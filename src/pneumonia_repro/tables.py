"""Generate summary tables for baseline characteristics, missingness, univariable analysis, and collinearity.

The tables draw on cells 4–7 of the preprocessing notebook, cells 2–7 of the p
notebook, and cells 12, 14, and 15 of the main notebook. Cell indices start at 0.
The module exports summary statistics.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "day_30mortality"
CLINICALLY_REMOVED = ["ALB", "Glu", "TBIL", "DBIL", "AST", "PT", "APTT"]
# The fixed order follows the clinical candidate predictors in cell 17 of the
# main notebook. Recomputed screening results are exported separately for comparison.
RANKING_CANDIDATES = [
    "age", "sex", "immunecompromised", "confusion", "fever", "RR", "HR",
    "O_PSI", "Neu", "Lym", "Hb", "CV", "CRP", "eGFR_new", "ADL_admission", "DM",
]
SCREENING_ORDER = [
    "age", "sex", "BMI", "Never_smoker", "Current_smoker", "Former_smoker",
    "COPD", "hypertension", "DM", "CAD", "cerobrovascular", "liver_disease",
    "immunecompromised", "confusion", "fever", "RR", "HR", "SBP", "DBP", "O_PSI",
    "WBC", "Neu", "Lym", "Mono", "Hb", "CV", "CRP", "Urea", "ALB", "Glu",
    "eGFR_new", "K", "Na", "TBIL", "DBIL", "ALT", "AST", "PT", "APTT", "Ddimer",
    "hsTNI", "ADL_admission",
]

# Source field, display name, variable type, category level, and decimal places.
BASELINE = [
    ("age", "Age, years", "continuous", None, 1),
    ("sex", "Sex", "categorical", {1: "Men", 0: "Women"}, 0),
    ("BMI", "BMI, kg/m2", "continuous", None, 1),
    ("smoking", "Cigarette smoking", "categorical", {0: "Never", 1: "Current", 2: "Former"}, 0),
    ("respiratory_disease", "Chronic pulmonary disease", "binary", 1, 0),
    ("hypertension", "Hypertension", "binary", 1, 0),
    ("DM", "Diabetes mellitus", "binary", 1, 0),
    ("CAD", "Coronary heart disease", "binary", 1, 0),
    ("cerobrovascular", "Cerebrovascular disease", "binary", 1, 0),
    ("CKD", "Chronic kidney disease", "binary", 1, 0),
    ("liver_disease", "Chronic liver disease", "binary", 1, 0),
    ("tumor", "Malignancy", "binary", 1, 0),
    ("immunecompromised", "Immunosuppression", "binary", 1, 0),
    ("confusion", "Confusion", "binary", 1, 0),
    ("fever", "Fever", "binary", 1, 0),
    ("RR", "Respiratory rate, breaths/min", "continuous", None, 1),
    ("HR", "Heart rate, beats/min", "continuous", None, 1),
    ("SBP", "Systolic blood pressure, mmHg", "continuous", None, 0),
    ("DBP", "Diastolic blood pressure, mmHg", "continuous", None, 0),
    ("O_PSI", "Severe hypoxia", "binary", 1, 0),
    ("WBC", "White blood cells, 10^9/L", "continuous", None, 1),
    ("CRP", "C-reactive protein, mg/L", "continuous", None, 1),
    ("Urea", "Urea, mmol/L", "continuous", None, 1),
    ("ALB", "Albumin, g/L", "continuous", None, 1),
    ("ADL_admission", "Activity of daily living", "continuous", None, 1),
    ("Oxygen", "Requiring oxygen therapy", "binary", 0, 0),
    ("nasal_mask", "Nasal cannula or oxygen mask", "binary", 1, 0),
    ("HFNC", "High flow nasal cannula", "binary", 1, 0),
    ("NIV", "Non-invasive mechanical ventilation", "binary", 1, 0),
    ("IMV", "Invasive mechanical ventilation", "binary", 1, 0),
    ("septicshock", "Septic shock", "binary", 1, 0),
    ("hospital_stay", "Length of hospital stay, days", "continuous", None, 1),
    ("ICU", "ICU admission", "binary", 1, 0),
    (TARGET, "30-day mortality", "binary", 1, 0),
    ("death", "In-hospital mortality", "binary", 1, 0),
]


def _baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "sex" in frame:
        frame["sex"] = frame["sex"].replace({2: 0})
    if "smoking" in frame:
        coding = {"0 从未吸烟": 0, "1 目前吸烟": 1, "2 已戒烟": 2}
        frame["smoking"] = frame["smoking"].map(lambda value: coding.get(value, value))
    if "BMI" not in frame and {"weight", "height"}.issubset(frame):
        height = pd.to_numeric(frame["height"], errors="coerce")
        frame["BMI"] = pd.to_numeric(frame["weight"], errors="coerce") / height.where(height > 0).pow(2) * 10000
    if "death" not in frame and "discharge" in frame:
        # Keep the original SPSS string codes and preserve missing discharge status.
        s = frame["discharge"]
        frame["death"] = s.eq("3 死亡").astype(float).where(s.notna())
    return frame


def _pvalue(a: pd.Series, b: pd.Series, continuous: bool) -> float:
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b):
        return np.nan
    if continuous:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    levels = sorted(set(a) | set(b))
    if len(levels) < 2:
        return np.nan
    # The original p.ipynb sets correction=False.
    counts = np.array([[int((s == level).sum()) for level in levels] for s in (a, b)])
    return float(chi2_contingency(counts, correction=False).pvalue)


def baseline_table(a: pd.DataFrame, b: pd.DataFrame, labels=("Retrospective", "Prospective"), *, omit_mortality=False) -> pd.DataFrame:
    """Report medians and quartiles or counts and percentages of the full cohort.

    Tests use the observed values. Fields absent from the input are marked as unavailable.
    """
    a, b = _baseline_frame(a), _baseline_frame(b)
    rows = []
    for field, label, kind, levels, decimals in BASELINE:
        if omit_mortality and field in {TARGET, "death"}:
            continue
        available = field in a and field in b
        p = _pvalue(a[field], b[field], kind == "continuous") if available else np.nan
        categories = levels.items() if kind == "categorical" else [(levels, "")]
        for j, (level, category_label) in enumerate(categories):
            row = {"Variable": field, "Label": label, "Category": category_label, "P": p if j == 0 else np.nan}
            for df, group in zip((a, b), labels):
                row[f"{group}_N"] = len(df)
                if field not in df:
                    row[group], row[f"{group}_observed"] = "UNAVAILABLE", 0
                else:
                    s = pd.to_numeric(df[field], errors="coerce").dropna()
                    row[f"{group}_observed"] = len(s)
                    if kind == "continuous":
                        if not len(s):
                            row[group] = "NA"
                        else:
                            q1, med, q3 = s.quantile([.25, .5, .75])
                            # Round NumPy scalars before formatting, as in the original analysis.
                            # For example, a median of 17.85 is displayed as 17.8.
                            q1, med, q3 = (np.round(value, decimals) for value in (q1, med, q3))
                            row[group] = f"{med:.{decimals}f} ({q1:.{decimals}f}-{q3:.{decimals}f})"
                    else:
                        n = int((s == level).sum())
                        row[group] = f"{n} ({100*n/len(df):.1f}%)" if len(df) else "NA"
            rows.append(row)
    return pd.DataFrame(rows)


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    return pd.DataFrame({"Variable": frame.columns, "N": n,
                         "Observed": frame.notna().sum().to_numpy(),
                         "Missing": frame.isna().sum().to_numpy(),
                         "Missing_percent": frame.isna().mean().to_numpy()*100})


def univariate_logistic(train: pd.DataFrame, predictors: list[str] | None = None) -> pd.DataFrame:
    if predictors is None:
        predictors = [x for x in SCREENING_ORDER if x in train]
    rows = []
    for field in predictors:
        df = train[[field, TARGET]].dropna().astype(float)
        row = {"Variable": field, "N": len(df), "Status": "ok"}
        if df[field].nunique() < 2:
            row.update(Status="constant", P=np.nan)
        else:
            try:
                with warnings.catch_warnings(record=True) as messages:
                    fit = sm.Logit(df[TARGET], sm.add_constant(df[[field]])).fit(disp=False)
                low, high = fit.conf_int().loc[field]
                row.update(OR=np.exp(fit.params[field]), CI_lower=np.exp(low), CI_upper=np.exp(high), P=fit.pvalues[field])
                if not fit.mle_retvals.get("converged", False) or messages:
                    row["Status"] = "warning_check_fit"
            except (ValueError, np.linalg.LinAlgError):
                row.update(Status="fit_failed", P=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def collinearity_table(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(columns=["Variable", "B", "Std. Error", "Standardized Beta", "t", "P", "Tolerance", "VIF"])
    X = sm.add_constant(train[features].astype(float), has_constant="add")
    y = train[TARGET].astype(float)
    model = sm.OLS(y, X).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vifs = [np.nan] + [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]
    beta = [np.nan] + [model.params[f] * np.std(train[f]) / np.std(y) for f in features]
    table = pd.DataFrame({"Variable": X.columns, "B": model.params.to_numpy(),
                          "Std. Error": model.bse.to_numpy(), "Standardized Beta": beta,
                          "t": model.tvalues.to_numpy(), "P": model.pvalues.to_numpy(),
                          "Tolerance": np.divide(1.0, np.asarray(vifs)), "VIF": vifs})
    return table.round(4)


def screening_tables(train: pd.DataFrame, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    single = univariate_logistic(train)
    single.to_csv(outdir / "table_s4_univariate.csv", index=False)
    # The original notebook screens on P values formatted to three decimal places.
    formatted_p = single.P.map(lambda x: float(f"{x:.3f}") if pd.notna(x) else np.nan)
    significant = single.loc[formatted_p < .05, "Variable"].tolist()
    first = collinearity_table(train, significant)
    first.to_csv(outdir / "table_s5_collinearity.csv", index=False)
    second_features = [f for f in significant if f != "WBC"]
    second = collinearity_table(train, second_features)
    second.to_csv(outdir / "table_s6_collinearity_without_wbc.csv", index=False)
    return {"significant_predictors": significant, "removed_for_collinearity": ["WBC"] if "WBC" in significant else [],
            "clinical_exclusions": CLINICALLY_REMOVED,
            "derived_ranking_candidates": [f for f in second_features if f not in CLINICALLY_REMOVED],
            "historical_ranking_candidates": RANKING_CANDIDATES,
            "note": "Feature ranking uses the manuscript candidate set. Screening results are also recorded."}


def cohort_tables(retrospective: pd.DataFrame, prospective: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_table(retrospective, prospective).to_csv(outdir / "table_1_baseline.csv", index=False)
    missingness_table(retrospective).to_csv(outdir / "table_s1_missingness.csv", index=False)
    missingness_table(prospective).to_csv(outdir / "table_s2_missingness.csv", index=False)
    baseline_table(retrospective.loc[retrospective[TARGET] == 0], retrospective.loc[retrospective[TARGET] == 1],
                   labels=("Survivors", "Non_survivors"), omit_mortality=True).to_csv(outdir / "table_s3_survival.csv", index=False)
