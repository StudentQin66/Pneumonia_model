"""Summarize model performance, compare clinical scores, and plot results using the original methods."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, fbeta_score, roc_auc_score, roc_curve
from .modeling import TARGET, FINAL_FEATURES, best_f2_threshold

DISPLAY_FEATURES = {"ADL_admission": "ADL", "Neu": "Neutrophils", "O_PSI": "Severe hypoxia", "immunecompromised": "Immune status", "Hb": "Haemoglobin", "eGFR_new": "eGFR", "Lym": "Lymphocytes"}
SCORE_COLUMNS = {"CURB-65": "CURB_65", "ADROP": "ADROP", "SMART-COP": "SMART_COP", "PSI": "PSI"}
COLORS = {"LR": "#E63946", "RF": "#2A9D8F", "SVM": "#E9C46A", "XGB": "#F4A261", "LGBM": "#264653", "GBDT": "#8AB17D", "BPNN": "#B56576", "CURB-65": "#2ca02c", "ADROP": "#1f77b4", "SMART-COP": "#ff7f0e", "PSI": "#B56576"}


def format_ci(value, se):
    return f"{value:.3f} ({max(0., value - 1.96 * se):.3f}-{min(1., value + 1.96 * se):.3f})"


def auc_interval(y, probabilities):
    y = np.asarray(y, dtype=int)
    auc = roc_auc_score(y, probabilities)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    se = np.sqrt(max(0, variance))
    return auc, max(0, auc - 1.96 * se), min(1, auc + 1.96 * se), se


def evaluate_performance(y_true, y_scores, threshold, model_name):
    """Use the original confidence interval methods, including binomial approximations for AP and F2."""
    y_true, y_scores = np.asarray(y_true, dtype=int), np.asarray(y_scores, dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Evaluation requires both outcome classes.")
    y_pred = (y_scores >= threshold).astype(int)
    auc, low, high, se_auc = auc_interval(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    values = {
        "AUROC": (auc, se_auc),
        "AUPRC": (average_precision_score(y_true, y_scores), None),
        "F2_Score": (fbeta_score(y_true, y_pred, beta=2, zero_division=0), None),
        "Accuracy": ((tp + tn) / n, None),
        "Sensitivity": (tp / (tp + fn), None),
        "Specificity": (tn / (tn + fp), None),
        "PPV": (tp / (tp + fp) if tp + fp else 0., None),
        "NPV": (tn / (tn + fn) if tn + fn else 0., None),
    }
    denominators = {"Sensitivity": tp + fn, "Specificity": tn + fp, "PPV": tp + fp, "NPV": tn + fn}
    row = {"Model": model_name, "Threshold": float(threshold), "N": n, "Deaths": int(y_true.sum())}
    for metric, (value, se) in values.items():
        denominator = denominators.get(metric, n)
        if se is None:
            se = np.sqrt(value * (1 - value) / denominator) if denominator else 0.
        row[metric] = format_ci(value, se)
        row[metric + "_estimate"] = float(value)
    row["Brier_score"] = brier_score_loss(y_true, y_scores)
    return row


def metrics_for_split(frame, predictions, thresholds):
    return pd.DataFrame([evaluate_performance(frame[TARGET], p, thresholds[name], name) for name, p in predictions.items()])


def compare_scores(train, test, lgbm_probabilities, lgbm_threshold):
    """Fit the logistic probability mapping and F2 threshold for each score on the training set only.

    Return a summary metrics table and probability vectors in memory.
    ROC plots and paired ΔAUROC analyses use the original clinical scores.
    Decision curves, calibration curves, and probability-based performance tables
    use the mapped probabilities.
    """
    y = train[TARGET].to_numpy(dtype=int)
    predictions = {"LGBM": np.asarray(lgbm_probabilities)}
    thresholds = {"LGBM": lgbm_threshold}
    mapping_rows = []
    for label, column in SCORE_COLUMNS.items():
        train_scores, test_scores = train[[column]], test[[column]]
        if train_scores.isna().any().any() or test_scores.isna().any().any():
            raise ValueError(f"{column} must be complete; traditional scores are excluded from imputation.")
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(train_scores, y)
        train_probabilities = model.predict_proba(train_scores)[:, 1]
        predictions[label] = model.predict_proba(test_scores)[:, 1]
        thresholds[label] = best_f2_threshold(y, train_probabilities)
        mapping_rows.append({"Score": label, "Intercept": float(model.intercept_[0]), "Coefficient": float(model.coef_[0, 0]), "Training_F2_threshold": thresholds[label]})
    return metrics_for_split(test, predictions, thresholds), predictions, pd.DataFrame(mapping_rows)


def calculate_dca(y, probabilities, thresholds):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    return np.array([((y == 1) & (probabilities >= t)).sum() / len(y) - ((y == 0) & (probabilities >= t)).sum() / len(y) * t / (1 - t) for t in thresholds])


def hosmer_lemeshow(y, probabilities, groups=10):
    data = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(probabilities)})
    data["group"] = pd.qcut(data["p"], q=groups, duplicates="drop")
    table = data.groupby("group", observed=False).agg(observed=("y", "sum"), expected=("p", "sum"), total=("y", "count"))
    terms = (table.observed - table.expected)**2 / table.expected.replace(0, np.nan)
    terms += ((table.total - table.observed) - (table.total - table.expected))**2 / (table.total - table.expected).replace(0, np.nan)
    stat = float(np.nansum(terms))
    df = max(len(table) - 2, 1)
    return {"Statistic": stat, "Degrees_of_freedom": df, "P_value": float(chi2.sf(stat, df)), "Groups": len(table)}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_curves(y, predictions, outdir, stem, *, roc_scores=None, title=None, calibration_bins=None, dca_range=(0.01, 0.40)):
    """Plot ROC, decision, and calibration curves and save their summary coordinates."""
    plt = _pyplot()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    y = np.asarray(y, dtype=int)
    threshold_grid = np.linspace(*dca_range, 100)
    if calibration_bins is None:
        calibration_bins = 8 if roc_scores is not None else 6
    rows, summaries = [], []
    for name, probabilities in predictions.items():
        scores = probabilities if roc_scores is None else roc_scores[name]
        fpr, tpr, _ = roc_curve(y, scores)
        auc, low, high, _ = auc_interval(y, scores)
        color = COLORS.get(name)
        axes[0].plot(fpr, tpr, color=color, label=f"{name}: {auc:.3f} ({low:.3f}-{high:.3f})")
        net = calculate_dca(y, probabilities, threshold_grid)
        area = float(np.trapezoid(net, threshold_grid))
        axes[1].plot(threshold_grid, net, color=color, label=f"{name}: AUDC {area:.3f}")
        observed, predicted = calibration_curve(y, probabilities, n_bins=calibration_bins, strategy="quantile")
        brier = brier_score_loss(y, probabilities)
        axes[2].plot(predicted, observed, "o-", color=color, label=f"{name}: Brier {brier:.3f}")
        summaries.append({"Model": name, "AUROC": auc, "AUROC_CI_low": low, "AUROC_CI_high": high, "AUDC": area, "AUDC_threshold_min": dca_range[0], "AUDC_threshold_max": dca_range[1], "Brier": brier})
        rows.extend({"Model": name, "Curve": "ROC", "x": x, "y": v} for x, v in zip(fpr, tpr))
        rows.extend({"Model": name, "Curve": "DCA", "x": x, "y": v} for x, v in zip(threshold_grid, net))
        rows.extend({"Model": name, "Curve": "Calibration", "x": x, "y": v} for x, v in zip(predicted, observed))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="(A) ROC", xlim=(0, 1), ylim=(0, 1))
    prevalence = y.mean()
    axes[1].plot(threshold_grid, prevalence - (1 - prevalence) * threshold_grid / (1 - threshold_grid), "k--", label="Treat all")
    axes[1].axhline(0, color="black", lw=1, label="Treat none")
    axes[1].set(xlabel="Threshold probability", ylabel="Net benefit", title="(B) Decision curves", xlim=(0, .4), ylim=(-.02, max(.1, prevalence + .02)))
    if dca_range[1] > .4:
        axes[1].set(xlim=(-.05, 1.05), ylim=(-.03, max(.08, prevalence + .01)))
    axes[2].plot([0, .4], [0, .4], "k--", lw=1)
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed event fraction", title="(C) Calibration", xlim=(-.02, .42), ylim=(-.02, .42))
    if roc_scores is not None and "LGBM" in predictions:
        hl = hosmer_lemeshow(y, predictions["LGBM"])
        axes[2].text(.52, .38, f"LGBM HL: {hl['Statistic']:.2f}, P = {hl['P_value']:.3f}", transform=axes[2].transAxes, fontsize=8)
    for ax in axes:
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outdir / (stem + ".png"), dpi=200)
    fig.savefig(outdir / (stem + ".pdf"))
    if stem == "figure_s2_s3_training":
        # Reuse the decision and calibration panels for supplementary figure
        # exports and vector PDFs.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for axis, panel_stem in [(axes[1], "figure_s2_training_dca"), (axes[2], "figure_s3_training_calibration")]:
            bbox = axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.06)
            fig.savefig(outdir / (panel_stem + ".png"), dpi=200, bbox_inches=bbox)
            fig.savefig(outdir / (panel_stem + ".pdf"), bbox_inches=bbox)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / (stem + "_curves.csv"), index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(outdir / (stem + "_summary.csv"), index=False)
    return summary


def plot_feature_sets(results, outdir):
    plt = _pyplot()
    outdir = Path(outdir)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for ax, (name, part) in zip(axes.flat, results.groupby("Model", sort=False)):
        part = part.sort_values("Num_Features")
        x = part.Num_Features.to_numpy()
        ax.plot(x, part.AUPRC_Mean, marker="o", color=COLORS[name], label=name)
        ax.fill_between(x, part.AUPRC_Lower, part.AUPRC_Upper, alpha=.2, color=COLORS[name])
        ax.set(title=name, xlabel="Number of predictors", ylabel="Bootstrap mean AUPRC")
    avg = results.groupby("Num_Features")["AUPRC_Mean"].mean()
    axes.flat[-1].plot(avg.index, avg.values, "o-", color="black")
    axes.flat[-1].set(title="Mean across algorithms", xlabel="Number of predictors", ylabel="Mean AUPRC")
    fig.tight_layout()
    fig.savefig(outdir / "figure_s1_feature_sets.png", dpi=200)
    fig.savefig(outdir / "figure_s1_feature_sets.pdf")
    plt.close(fig)


def explain_lgbm(model, test, outdir, *, allow_individual_plots=False):
    """Explain the tree model from the first calibration fold, as in cell 37.

    Mean absolute SHAP values summarize feature importance. Setting
    allow_individual_plots to True produces scatter and dependence plots of
    individual values. SHAP values represent contributions to the tree model's
    raw margin.
    """
    import shap
    plt = _pyplot()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fold = model.calibrated_classifiers_[0]
    pipe = getattr(fold, "estimator", getattr(fold, "base_estimator", None))
    features = list(model.feature_names_in_)
    X = test[features]
    scaled = pipe.named_steps["scaler"].transform(X)
    values = shap.TreeExplainer(pipe.named_steps["classifier"]).shap_values(scaled)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[1]
    if values.shape != X.shape:
        raise ValueError("Unexpected binary SHAP array shape.")
    importance = pd.DataFrame({"Feature": features, "Mean_absolute_SHAP_raw_margin": np.abs(values).mean(axis=0)}).sort_values("Mean_absolute_SHAP_raw_margin", ascending=False, kind="mergesort")
    importance.to_csv(outdir / "figure_5_shap_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.Feature.map(DISPLAY_FEATURES)[::-1], importance.Mean_absolute_SHAP_raw_margin[::-1], color=COLORS["LGBM"])
    ax.set(xlabel="Mean absolute SHAP value (first fold tree raw margin)")
    fig.tight_layout()
    fig.savefig(outdir / "figure_5A_shap_importance.png", dpi=200)
    fig.savefig(outdir / "figure_5A_shap_importance.pdf")
    plt.close(fig)
    if allow_individual_plots:
        fig = plt.figure(figsize=(20, 14))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=.34,
                                left=.08, right=.96, bottom=.07, top=.94)
        top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=.50)
        bottom = outer[1].subgridspec(2, 4, wspace=.43, hspace=.44)
        ax_a = fig.add_subplot(top[0, 0])
        ordered_names = importance.Feature.map(DISPLAY_FEATURES)
        ax_a.barh(ordered_names[::-1], importance.Mean_absolute_SHAP_raw_margin[::-1], color="#1e88e5")
        ax_a.set_title("(A) First-fold tree SHAP importance", fontsize=13, pad=12)
        ax_a.set_xlabel("Mean absolute SHAP value (raw margin)", fontsize=11)
        ax_a.tick_params(axis="both", labelsize=10)
        ax_a.set_xlim(0, max(importance.Mean_absolute_SHAP_raw_margin.max() * 1.12, 1e-6))
        ax_b = fig.add_subplot(top[0, 1])
        X_plot = X.rename(columns=DISPLAY_FEATURES)
        explanation = shap.Explanation(values=values, data=X_plot.to_numpy(), feature_names=X_plot.columns.tolist())
        shap.plots.beeswarm(explanation, ax=ax_b, show=False, plot_size=None,
                            max_display=len(features), color_bar=True, s=12)
        ax_b.set_title("(B) First-fold tree SHAP summary", fontsize=13, pad=12)
        ax_b.set_xlabel("SHAP value (raw margin)", fontsize=11)
        ax_b.tick_params(axis="both", labelsize=10)
        for j, feature in enumerate(importance.Feature):
            ax = fig.add_subplot(bottom[j // 4, j % 4])
            feature_index = features.index(feature)
            ax.scatter(X[feature], values[:, feature_index], s=13, alpha=.85, color="#1e88e5")
            ax.set_title(f"({chr(97 + j)}) {DISPLAY_FEATURES[feature]}", fontsize=12, pad=8)
            ax.set_xlabel(DISPLAY_FEATURES[feature], fontsize=10)
            ax.set_ylabel("SHAP value", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
        fig.text(.08, .018, "SHAP contributions explain the first calibration fold's tree raw margin; feature axes retain original units.", fontsize=10)
        fig.savefig(outdir / "figure_5_shap.png", dpi=200)
        fig.savefig(outdir / "figure_5_shap.pdf")
        plt.close(fig)
    return importance


SUBGROUPS = [
    ("Age", [("65-74 years", lambda d: d.age.between(65, 74)), (">=75 years", lambda d: d.age >= 75)]),
    ("Sex", [("Female", lambda d: d.sex == 0), ("Male", lambda d: d.sex == 1)]),
    ("Immune", [("Immunocompetent", lambda d: d.immunecompromised == 0), ("Immunocompromised", lambda d: d.immunecompromised == 1)]),
    ("PSI", [("I-III", lambda d: d.PSI <= 90), ("IV-V", lambda d: d.PSI > 90)]),
    ("CURB-65", [("0-1", lambda d: d.CURB_65.between(0, 1)), ("2-5", lambda d: d.CURB_65.between(2, 5))]),
    ("ADL", [(">40", lambda d: d.ADL_admission > 40), ("<=40", lambda d: d.ADL_admission <= 40)]),
]


def bootstrap_auc_matrix(y, scores, n_bootstraps, rng):
    """Run paired bootstrapping with equivalent multinomial weights for each individual."""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    n, m = scores.shape
    if n == 0 or len(np.unique(y)) != 2:
        return np.full(m, np.nan), np.full((n_bootstraps, m), np.nan)
    point = np.array([roc_auc_score(y, scores[:, j]) for j in range(m)])
    weights = rng.multinomial(n, np.full(n, 1. / n), size=n_bootstraps).astype(float)
    pos, neg = y == 1, y == 0
    wp, wn = weights[:, pos], weights[:, neg]
    denom = wp.sum(axis=1) * wn.sum(axis=1)
    valid = denom > 0
    boot = np.full((n_bootstraps, m), np.nan)
    for j in range(m):
        ps, ns = scores[pos, j], scores[neg, j]
        concordance = (ps[:, None] > ns[None, :]).astype(float) + .5 * (ps[:, None] == ns[None, :])
        numerators = np.einsum("bi,ij,bj->b", wp, concordance, wn, optimize=True)
        boot[valid, j] = numerators[valid] / denom[valid]
    return point, boot


def _percentile(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return tuple(np.quantile(finite, [.025, .975])) if len(finite) else (np.nan, np.nan)


def _bootstrap_p(values):
    values = np.asarray(values)[np.isfinite(values)]
    if not len(values):
        return np.nan
    return min(1., 2 * min(((values <= 0).sum() + 1) / (len(values) + 1), ((values >= 0).sum() + 1) / (len(values) + 1)))


def subgroup_analysis(test, lgbm_probabilities, outdir=None, *, n_bootstraps=2000, seed=20260816):
    """Compute paired ΔAUROC and bootstrap interaction tests as in the subgroup notebook.

    Resample each subgroup level independently, sharing individual resampling
    weights across paired model predictions within that level. Subtract the
    difference vectors of the two independent levels to assess interaction.
    Synthetic levels with too few observations or insufficient outcome classes
    are marked as unavailable.
    """
    columns = ["PSI", "CURB_65", "ADROP", "SMART_COP"]
    labels = ["PSI", "CURB-65", "ADROP", "SMART-COP"]
    scores = np.column_stack([lgbm_probabilities, test[columns].to_numpy()])
    y = test[TARGET].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    records, interactions = [], []
    for subgroup, levels in SUBGROUPS:
        parts = []
        for level, masker in levels:
            mask = masker(test).fillna(False).to_numpy(dtype=bool)
            part_y, part_scores = y[mask], scores[mask]
            valid = np.isfinite(part_scores).all(axis=1)
            part_y, part_scores = part_y[valid], part_scores[valid]
            point, boot = bootstrap_auc_matrix(part_y, part_scores, n_bootstraps, rng)
            deltas = []
            for j, label in enumerate(labels, 1):
                delta, delta_boot = point[0] - point[j], boot[:, 0] - boot[:, j]
                low, high = _percentile(delta_boot)
                records.append({"subgroup": subgroup, "level": level, "comparator": label, "n": len(part_y), "deaths": int(part_y.sum()), "mortality_percent": 100 * part_y.mean() if len(part_y) else np.nan, "lgbm_auc": point[0], "traditional_auc": point[j], "delta_auroc": delta, "ci_lower": low, "ci_upper": high, "valid_bootstraps": int(np.isfinite(delta_boot).sum())})
                deltas.append((delta, delta_boot))
            parts.append(deltas)
        for j, label in enumerate(labels):
            diff = parts[0][j][0] - parts[1][j][0]
            boots = parts[0][j][1] - parts[1][j][1]
            low, high = _percentile(boots)
            interactions.append({"subgroup": subgroup, "comparator": label, "level_1": levels[0][0], "level_2": levels[1][0], "interaction_difference": diff, "interaction_ci_lower": low, "interaction_ci_upper": high, "p_interaction": _bootstrap_p(boots), "valid_bootstraps": int(np.isfinite(boots).sum())})
    # As in the original analysis, restart from the same seed for the overall comparison.
    point, boot = bootstrap_auc_matrix(y, scores, n_bootstraps, np.random.default_rng(seed))
    overall = []
    for j, label in enumerate(labels, 1):
        delta_boot = boot[:, 0] - boot[:, j]
        low, high = _percentile(delta_boot)
        overall.append({"subgroup": "All", "level": "All", "comparator": label, "n": len(y), "deaths": int(y.sum()), "mortality_percent": 100 * y.mean(), "lgbm_auc": point[0], "traditional_auc": point[j], "delta_auroc": point[0] - point[j], "ci_lower": low, "ci_upper": high, "valid_bootstraps": int(np.isfinite(delta_boot).sum())})
    levels_df, interactions_df = pd.DataFrame(overall + records), pd.DataFrame(interactions)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        levels_df.to_csv(outdir / "figure_4_subgroup_delta_auroc.csv", index=False)
        interactions_df.to_csv(outdir / "figure_4_subgroup_interaction_tests.csv", index=False)
        _plot_subgroups(levels_df, interactions_df, outdir)
    return levels_df, interactions_df


def _plot_subgroups(levels, interactions, outdir):
    """Reproduce the layout with sample sizes, forest plots for four comparator scores, and interaction P values."""
    plt = _pyplot()
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(1, 5, width_ratios=[4.2, 4, 4, 4, 4], left=.025, right=.99, bottom=.14, top=.87, wspace=.16)
    count_ax = fig.add_subplot(grid[0, 0])
    row_keys = [("All", "All")]
    positions = [18.6]
    midpoints = {}
    y = 17
    for subgroup, parts in SUBGROUPS:
        midpoints[subgroup] = y - .5
        for j, (level, _) in enumerate(parts):
            row_keys.append((subgroup, level))
            positions.append(y - j)
        y -= 3
    counts = levels.drop_duplicates(["subgroup", "level"]).set_index(["subgroup", "level"])
    count_ax.set(xlim=(0, 1), ylim=(.3, 19.4))
    count_ax.axis("off")
    count_ax.text(0, 1.025, "Subgroup", transform=count_ax.transAxes, fontsize=10, fontweight="bold")
    count_ax.text(.63, 1.025, "N", ha="center", transform=count_ax.transAxes, fontsize=10, fontweight="bold")
    count_ax.text(.88, 1.025, "Mortality n (%)", ha="center", transform=count_ax.transAxes, fontsize=10, fontweight="bold")
    for key, y in zip(row_keys, positions):
        row = counts.loc[key]
        if key[0] != "All" and key[1] == dict(SUBGROUPS)[key[0]][0][0]:
            count_ax.text(0, y + .6, key[0], fontsize=10, fontweight="bold")
        count_ax.text(.02, y, key[1], va="center", fontsize=9)
        count_ax.text(.63, y, str(int(row.n)), va="center", ha="center", fontsize=9)
        count_ax.text(.88, y, f"{int(row.deaths)} ({row.mortality_percent:.1f})", va="center", ha="center", fontsize=9)
    finite_low = levels.ci_lower.dropna()
    finite_high = levels.ci_upper.dropna()
    low = min(-.05, finite_low.min() if len(finite_low) else -.05)
    high = max(.05, finite_high.max() if len(finite_high) else .05)
    low, high = np.floor((low - .02) / .05) * .05, np.ceil((high + .02) / .05) * .05
    width = high - low
    est_x, separator = high + .10 * width, high + .03 * width
    def number(v):
        if not np.isfinite(v):
            return "NA"
        return f"{0. if abs(v) < .005 else v:.2f}"
    for panel, comparator in enumerate(["PSI", "CURB-65", "ADROP", "SMART-COP"], 1):
        inside = grid[0, panel].subgridspec(1, 2, width_ratios=[3.25, .75], wspace=.04)
        ax = fig.add_subplot(inside[0, 0])
        p_ax = fig.add_subplot(inside[0, 1], sharey=ax)
        ax.set(xlim=(low, high + 1.2 * width), ylim=(.3, 19.4))
        p_ax.set(xlim=(0, 1), ylim=(.3, 19.4))
        p_ax.axis("off")
        part = levels[levels.comparator == comparator].set_index(["subgroup", "level"])
        for key, y in zip(row_keys, positions):
            row = part.loc[key]
            if np.isfinite(row.delta_auroc):
                ax.plot([row.ci_lower, row.ci_upper], [y, y], color="#484878", lw=1.2)
                ax.plot(row.delta_auroc, y, "o", color="#0F4D92", markersize=4)
            ax.text(est_x, y, f"{number(row.delta_auroc)} ({number(row.ci_lower)}, {number(row.ci_upper)})", va="center", fontsize=9)
        pi = interactions[interactions.comparator == comparator].set_index("subgroup")
        p_ax.text(.62, positions[0], "-", ha="center", va="center", fontsize=9)
        for subgroup, midpoint in midpoints.items():
            p = pi.loc[subgroup, "p_interaction"]
            text = "NA" if not np.isfinite(p) else ("<0.001" if p < .001 else f"{p:.3f}")
            p_ax.plot([.10, .20, .20, .10], [midpoint + .5, midpoint + .5, midpoint - .5, midpoint - .5], color="#A8A8A8", lw=.7)
            p_ax.text(.62, midpoint, text, ha="center", va="center", fontsize=9)
        for a in [ax, p_ax, count_ax]:
            a.axhspan(18.15, 19.05, color="#F5F7FB", zorder=0)
            for midpoint in midpoints.values():
                a.axhspan(midpoint - .95, midpoint + .95, color="#F5F7FB", zorder=0)
        ax.axvline(0, color="#9c9c9c", ls="--", lw=.9)
        ax.axvline(separator, color="#d6d6d6", lw=.7)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(low, high)
        from matplotlib.ticker import MaxNLocator
        ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
        ticks = ticks[(ticks >= low) & (ticks <= high)]
        ax.set_xticks(ticks)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title("Delta AUROC (95% CI)", fontsize=10, fontweight="bold", pad=12)
        p_ax.set_title("P for\ninteraction", fontsize=9, fontweight="bold", pad=5)
        left, right = ax.get_position().x0, p_ax.get_position().x1
        fig.text((left + right) / 2, .925, f"LGBM vs {comparator}", ha="center", fontsize=12, fontweight="bold")
    fig.suptitle("Subgroup comparison of LGBM and traditional pneumonia scores", fontsize=16, fontweight="bold", y=.98)
    fig.text(.03, .04, "Delta AUROC = AUROC(LGBM) - AUROC(conventional score). Positive values favor LGBM.\nPaired patient bootstrap 95% CIs; interaction P values are unadjusted. N is the number of patients; mortality is n (%).", fontsize=10)
    fig.savefig(outdir / "figure_4_subgroups.png", dpi=200)
    fig.savefig(outdir / "figure_4_subgroups.pdf")
    plt.close(fig)
