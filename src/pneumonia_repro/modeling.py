"""Reproduce model training and feature selection from the final analysis notebook.

As in the original workflow, randomized search precedes calculation of the
reported out-of-fold predictions. The out-of-fold procedure nests the fitting
steps and leaves feature selection and hyperparameter search outside those folds.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "day_30mortality"
CANDIDATE_FEATURES = ["age", "sex", "immunecompromised", "confusion", "fever", "RR", "HR", "O_PSI", "Neu", "Lym", "Hb", "CV", "CRP", "eGFR_new", "ADL_admission", "DM"]
FINAL_FEATURES = ["ADL_admission", "Neu", "O_PSI", "immunecompromised", "Hb", "eGFR_new", "Lym"]
MODEL_NAMES = ("LR", "RF", "SVM", "XGB", "LGBM", "GBDT", "BPNN")


def model_specs(y, seed=42, n_jobs=1):
    """Define the seven models and full randomized search spaces from cell 22."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if (counts == 0).any():
        raise ValueError("Training requires both outcome classes.")
    spw = counts[0] / counts[1]
    specs = {
        "LR": (LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed), {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"], "solver": ["liblinear", "saga"]}),
        "RF": (RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs), {"n_estimators": [50, 100, 200, 300], "max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None]}),
        "SVM": (SVC_factory(seed), {"C": [0.01, 0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01, 0.1], "kernel": ["rbf", "linear"]}),
        "XGB": (XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=n_jobs), {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 5, 7, 9], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]}),
        "LGBM": (LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=n_jobs, verbosity=-1), {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "num_leaves": [15, 31, 63, 127], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [10, 20, 30]}),
        "GBDT": (GradientBoostingClassifier(random_state=seed), {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 5, 7], "subsample": [0.6, 0.8, 1.0], "min_samples_split": [2, 5, 10]}),
        "BPNN": (MLPClassifier(max_iter=2000, early_stopping=True, random_state=seed), {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)], "alpha": [0.0001, 0.001, 0.01, 0.1], "activation": ["relu", "tanh"], "learning_rate_init": [0.001, 0.01, 0.05]}),
    }
    return {name: (est, {"classifier__" + k: v for k, v in space.items()}) for name, (est, space) in specs.items()}


def SVC_factory(seed):
    from sklearn.svm import SVC
    return SVC(class_weight="balanced", probability=True, random_state=seed)


def _cv(y, folds, seed):
    if np.bincount(np.asarray(y, dtype=int), minlength=2).min() < folds:
        raise ValueError(f"Each outcome class requires at least {folds} observations.")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("classifier", clone(estimator))])


def best_f2_threshold(y, probabilities):
    p, r, thresholds = precision_recall_curve(y, probabilities)
    f2 = np.divide(5 * p[:-1] * r[:-1], 4 * p[:-1] + r[:-1], out=np.zeros(len(thresholds)), where=(4 * p[:-1] + r[:-1]) != 0)
    return float(thresholds[np.argmax(f2)])


def rank_features(train, *, seed=42, folds=5, n_iter=5, n_jobs=1, model_names=MODEL_NAMES, permutation_repeats=5):
    """Rank the 16 prespecified clinical predictors by summed ranks across seven algorithms."""
    X, y = train[CANDIDATE_FEATURES], train[TARGET].to_numpy(dtype=int)
    cv = _cv(y, folds, seed)
    result = pd.DataFrame({"Feature": CANDIDATE_FEATURES})
    tuned = {}
    for name, (est, params) in model_specs(y, seed, n_jobs).items():
        if name not in model_names:
            continue
        search = RandomizedSearchCV(_pipeline(est), params, n_iter=n_iter, scoring="average_precision", cv=cv, n_jobs=n_jobs, random_state=seed, error_score="raise")
        search.fit(X, y)
        fitted = search.best_estimator_.named_steps["classifier"]
        tuned[name] = clone(fitted)
        if hasattr(fitted, "feature_importances_"):
            scores = fitted.feature_importances_
        elif name == "LR":
            scores = np.abs(fitted.coef_[0])
        else:
            scores = np.maximum(permutation_importance(search.best_estimator_, X, y, scoring="average_precision", n_repeats=permutation_repeats, random_state=seed, n_jobs=n_jobs).importances_mean, 0)
        result[name + "_Rank"] = pd.Series(scores).rank(ascending=False, method="min")
    ranks = [c for c in result if c.endswith("_Rank")]
    if not ranks:
        raise ValueError("At least one model is required.")
    result["Sum_Ranks"] = result[ranks].sum(axis=1)
    result["Final_Order"] = result["Sum_Ranks"].rank(ascending=True, method="min")
    return result.sort_values("Final_Order", kind="quicksort").reset_index(drop=True), tuned


def bootstrap_auprc(y, probabilities, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) == 2:
            scores.append(average_precision_score(y[idx], probabilities[idx]))
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), *np.percentile(scores, [2.5, 97.5]).tolist()


def evaluate_feature_sets(train, ranking, tuned_estimators, *, seed=42, folds=5, n_bootstraps=1000, n_jobs=1):
    ordered = ranking["Feature"].tolist()
    y = train[TARGET].to_numpy(dtype=int)
    cv = _cv(y, folds, seed)
    records = []
    for k in range(1, len(ordered) + 1):
        for name, est in tuned_estimators.items():
            probabilities = cross_val_predict(_pipeline(est), train[ordered[:k]], y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
            mean, low, high = bootstrap_auprc(y, probabilities, n_bootstraps, seed)
            records.append({"Model": name, "Num_Features": k, "AUPRC_Mean": mean, "AUPRC_Lower": low, "AUPRC_Upper": high})
    return pd.DataFrame(records)


@dataclass
class ModelResults:
    models: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=lambda: {"train": {}, "val": {}, "test": {}})
    thresholds: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    features: list = field(default_factory=list)


def fit_models(train, val, test, *, features=FINAL_FEATURES, seed=42, folds=5, calibration_folds=5, n_iter=20, n_jobs=1, model_names=MODEL_NAMES):
    """Refit and calibrate seven models and retain thresholds from training out-of-fold predictions.

    Model fitting uses training outcomes only. This function returns models and
    individual predictions in memory.
    """
    y = train[TARGET].to_numpy(dtype=int)
    cv = _cv(y, folds, seed)
    # Each outer training fold must have enough cases for inner calibration.
    min_class = np.bincount(y, minlength=2).min()
    if min_class - int(np.ceil(min_class / folds)) < calibration_folds:
        raise ValueError("Too few minority-class observations for nested calibration; increase synthetic cohort size.")
    result = ModelResults(features=list(features))
    for name, (est, params) in model_specs(y, seed, n_jobs).items():
        if name not in model_names:
            continue
        search = RandomizedSearchCV(_pipeline(est), params, n_iter=n_iter, scoring="average_precision", cv=cv, n_jobs=n_jobs, random_state=seed, error_score="raise")
        search.fit(train[features], y)
        calibrated = CalibratedClassifierCV(estimator=search.best_estimator_, method="sigmoid", cv=calibration_folds, ensemble=True, n_jobs=n_jobs)
        calibrated.fit(train[features], y)
        oof = cross_val_predict(clone(calibrated), train[features], y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
        result.models[name] = calibrated
        result.predictions["train"][name] = oof
        result.predictions["val"][name] = calibrated.predict_proba(val[features])[:, 1]
        result.predictions["test"][name] = calibrated.predict_proba(test[features])[:, 1]
        result.thresholds[name] = best_f2_threshold(y, oof)
        result.best_params[name] = search.best_params_
    if not result.models:
        raise ValueError("At least one recognized model is required.")
    return result
