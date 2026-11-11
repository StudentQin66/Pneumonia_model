"""Command-line workflow for data processing, model fitting, and manuscript outputs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import TARGET, FINAL_FEATURES, prepare_data
from .evaluation import (SCORE_COLUMNS, compare_scores, explain_lgbm, metrics_for_split,
                         plot_curves, plot_feature_sets, subgroup_analysis)
from .modeling import fit_models, rank_features, evaluate_feature_sets
from .synthetic import generate_cohorts
from .tables import cohort_tables, screening_tables
from .missingness import manuscript_missingness_table

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value) -> None:
    """Write run settings and summary checks as UTF-8 JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def environment() -> dict:
    """Record the Python version, operating system, and analysis package versions."""
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "shap", "statsmodels", "matplotlib", "openpyxl"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "packages": versions}


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def safe_output_dir(directory: str | Path) -> Path:
    out = Path(directory).expanduser().resolve()
    if _inside(out, ROOT) and not _inside(out, ROOT / "outputs"):
        raise ValueError("Save in-repository results under outputs/.")
    if out.exists() and any(out.iterdir()):
        raise ValueError("The output directory must be empty. Use a separate directory for each run.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_flow(audit: dict, outdir: Path, synthetic: bool) -> None:
    """Draw the flowchart from the observed cohort counts using the manuscript's terminology."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = audit.get("counts", {})
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    nodes = [(.13, .80, f"Retrospective cohort\nN={counts.get('retrospective_raw', 'unavailable')}"),
             (.13, .48, f"Eligible development cohort\nN={counts.get('retrospective_eligible', 'unavailable')}"),
             (.43, .73, f"Training (80%)\nN={counts.get('train', 'unavailable')}"),
             (.43, .28, f"Validation (20%)\nN={counts.get('validation', 'unavailable')}"),
             (.78, .80, f"Prospective cohort\nN={counts.get('prospective_raw', 'unavailable')}"),
             (.78, .48, f"Eligible test cohort\nN={counts.get('test', 'unavailable')}"),
             (.78, .12, "Prospective evaluation\nScore comparison, subgroups, SHAP")]
    for x, y, text in nodes:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.65", facecolor="#edf2f7", edgecolor="#334155"))
    for xy, xytext in [((.13, .56), (.13, .72)), ((.34, .71), (.22, .49)), ((.34, .30), (.22, .47)),
                       ((.78, .56), (.78, .72)), ((.78, .22), (.78, .40))]:
        ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.text(.43, .50, "Fit preprocessing on training only\nScreening and ranking\nCalibrated seven-predictor models", ha="center", va="center", fontsize=9)
    ax.set_title("Synthetic demonstration flow" if synthetic else "Analysis cohort flow reconstructed from input counts", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "figure_1_flow.png", dpi=180)
    fig.savefig(outdir / "figure_1_flow.pdf")
    plt.close(fig)


def _frozen(args):
    # Read only the permitted analysis fields from previously imputed splits for
    # downstream checks. Use the original cohorts for full-cohort baseline and missingness tables.
    from .data import ALLOWED_INPUTS
    frames = [pd.read_csv(p, usecols=lambda c: c in ALLOWED_INPUTS) for p in [args.train, args.validation, args.test]]
    required = set(FINAL_FEATURES + [TARGET] + list(SCORE_COLUMNS.values()))
    for frame in frames:
        if frame.empty or not required.issubset(frame):
            raise ValueError("Previously imputed splits must include observed outcomes, the seven final predictors, and the four conventional scores.")
        if frame[list(required)].isna().any().any() or not frame[TARGET].isin([0, 1]).all():
            raise ValueError("Analysis fields in previously imputed splits must be complete, with outcomes coded as 0 or 1.")
    train, val, test = frames
    audit = {"input_stage": "frozen imputed splits; upstream preprocessing not revalidated by this route",
             "counts": {"train": len(train), "validation": len(val), "test": len(test)},
             "deaths": {"train": int(train[TARGET].sum()), "validation": int(val[TARGET].sum()), "test": int(test[TARGET].sum())}}
    return train, val, test, audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("--mode", choices=["synthetic", "real", "frozen"], default="synthetic", help="Input mode. Use synthetic for independently generated data, real for complete original cohorts, or frozen for previously imputed splits.")
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke", help="Use smoke for a quick demonstration or paper for the full manuscript settings.")
    parser.add_argument("--output-dir", default="outputs/smoke", help="Use a new or empty output directory. Save in-repository results under outputs/.")
    parser.add_argument("--retrospective", help="CSV or Excel file containing the complete retrospective cohort.")
    parser.add_argument("--prospective", help="CSV or Excel file containing the complete prospective cohort.")
    parser.add_argument("--train", help="Existing training split for frozen mode.")
    parser.add_argument("--validation", help="Existing validation split for frozen mode.")
    parser.add_argument("--test", help="Existing test split for frozen mode.")
    parser.add_argument("--prospective-outcome", choices=["death", "survival"], default="death", help="Outcome coding in the prospective input. Use death if 1 denotes death or survival if 1 denotes survival.")
    parser.add_argument("--egfr-formula", choices=["china2019", "ckdepi2021", "source_final"], default=None, help="eGFR calculation method. The manuscript uses china2019. The source_final option requires eGFR_new in the input.")
    parser.add_argument("--missingness-policy", choices=["manuscript", "legacy-complete-identifiers"], default="manuscript", help="Missingness calculation policy. The reported results use manuscript.")
    parser.add_argument("--n-retrospective", type=int, default=360, help="Number of synthetic retrospective records (default 360).")
    parser.add_argument("--n-prospective", type=int, default=240, help="Number of synthetic prospective records (default 240).")
    parser.add_argument("--synthetic-missing-rate", type=float, default=.03, help="Random missingness rate for synthetic candidate predictors (default 0.03).")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip Table S7 and Figure S1 when checking downstream analyses only.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP figure generation.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of workers. Defaults to the profile setting.")
    parser.add_argument("--models", nargs="+", choices=["LR", "RF", "SVM", "XGB", "LGBM", "GBDT", "BPNN"], help="Subset of models for a diagnostic run. Include LGBM.")
    args = parser.parse_args(argv)
    if args.mode == "synthetic" and any([args.retrospective, args.prospective, args.train, args.validation, args.test]):
        parser.error("Synthetic mode generates cohorts in memory. Omit the input file paths.")
    if args.mode == "real" and (not args.retrospective or not args.prospective):
        parser.error("Real mode requires both --retrospective and --prospective.")
    if args.mode == "frozen" and not all([args.train, args.validation, args.test]):
        parser.error("Frozen mode requires --train, --validation, and --test.")
    config = json.loads((ROOT / f"config/{args.profile}.json").read_text())
    if args.models:
        config["models"] = args.models
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    if "LGBM" not in config["models"]:
        parser.error("The score comparisons, subgroup analyses, and SHAP analyses require LGBM. Include it in --models.")
    out = safe_output_dir(args.output_dir)
    start = time.monotonic()
    write_json(out / "run_manifest.json", {"status": "started", "mode": args.mode, "profile": args.profile,
               "synthetic": args.mode == "synthetic", "configuration": config, "environment": environment()})
    print("Reading, checking, and processing the input cohorts.", flush=True)
    if args.mode == "frozen":
        train, val, test, audit = _frozen(args)
    else:
        sources = generate_cohorts(args.n_retrospective, args.n_prospective, config["seed"], args.synthetic_missing_rate) if args.mode == "synthetic" else (args.retrospective, args.prospective)
        data = prepare_data(*sources, prospective_outcome=args.prospective_outcome if args.mode == "real" else None,
                            egfr_formula=args.egfr_formula, missingness_policy=args.missingness_policy, seed=config["seed"])
        train, val, test, audit = data.train, data.val, data.test, data.audit
        cohort_tables(data.retrospective, data.prospective, out)
        manuscript_missingness_table(sources[0], canonical=data.retrospective).to_csv(out / "table_s1_missingness.csv", index=False)
        manuscript_missingness_table(sources[1], canonical=data.prospective).to_csv(out / "table_s2_missingness.csv", index=False)
    write_json(out / "cohort_audit.json", audit)
    plot_flow(audit, out, args.mode == "synthetic")
    screening = screening_tables(train, out)
    write_json(out / "screening_audit.json", screening)
    selection = {"method": "Manuscript predictor set (seven variables)", "features": FINAL_FEATURES}
    if not args.skip_feature_selection:
        print("Ranking predictors and evaluating cumulative predictor sets.", flush=True)
        ranking, tuned = rank_features(train, seed=config["seed"], folds=config["folds"], n_iter=config["ranking_search_iterations"],
                                      n_jobs=config["n_jobs"], model_names=config["models"], permutation_repeats=config["permutation_repeats"])
        ranking.to_csv(out / "table_s7_feature_ranking.csv", index=False)
        feature_sets = evaluate_feature_sets(train, ranking, tuned, seed=config["seed"], folds=config["folds"],
                                             n_bootstraps=config["feature_bootstraps"], n_jobs=config["n_jobs"])
        feature_sets.to_csv(out / "figure_s1_feature_sets.csv", index=False)
        plot_feature_sets(feature_sets, out)
        mean = feature_sets.groupby("Num_Features").AUPRC_Mean.mean()
        best_k = int(mean.idxmax())
        selection.update(recomputed_optimal_count=best_k, recomputed_predictors=ranking.Feature.head(best_k).tolist(),
                         ranked_top_seven=ranking.Feature.head(7).tolist(),
                         published_features_same_set=set(ranking.Feature.head(7)) == set(FINAL_FEATURES),
                         published_features_same_order=ranking.Feature.head(7).tolist() == FINAL_FEATURES,
                         optimal_count_matches_published=best_k == len(FINAL_FEATURES),
                         note="Final models use the manuscript predictor set. This file also records the predictors selected in the current run.")
    write_json(out / "feature_selection.json", selection)
    print("Tuning and calibrating models and generating out-of-fold training predictions.", flush=True)
    result = fit_models(train, val, test, features=FINAL_FEATURES, seed=config["seed"], folds=config["folds"],
                        calibration_folds=config["calibration_folds"], n_iter=config["final_search_iterations"],
                        n_jobs=config["n_jobs"], model_names=config["models"])
    write_json(out / "best_parameters.json", result.best_params)
    # The original holdout evaluation used training thresholds rounded to four decimal places.
    # One GBDT validation prediction is close to that threshold, so record both precisions for Table 2.
    heldout_thresholds = {name: round(value, config["heldout_threshold_decimals"]) for name, value in result.thresholds.items()}
    write_json(out / "training_thresholds.json", {"oof_full_precision": result.thresholds, "heldout_rounded": heldout_thresholds,
        "rounding_decimals": config["heldout_threshold_decimals"], "source": "main notebook training cell22 and validation cell29; prospective F2 script"})
    train_metrics = metrics_for_split(train, result.predictions["train"], result.thresholds)
    val_metrics = metrics_for_split(val, result.predictions["val"], heldout_thresholds)
    train_metrics.to_csv(out / "table_s8_training.csv", index=False)
    val_metrics.to_csv(out / "table_2_validation.csv", index=False)
    best_model = max(config["models"], key=lambda name: roc_auc_score(val[TARGET], result.predictions["val"][name]))
    write_json(out / "model_selection.json", {"selected_by_validation_auroc": best_model,
        "published_model": "LGBM", "agrees_with_manuscript": best_model == "LGBM",
        "test_labels_used_for_model_selection": False,
        "interpretation": "LGBM is used for the prospective score comparisons, subgroup analyses, and SHAP analysis."})
    print("Generating summary tables, evaluation curves, and subgroup results.", flush=True)
    plot_curves(val[TARGET], result.predictions["val"], out, "figure_2_validation", title="Validation set")
    plot_curves(train[TARGET], result.predictions["train"], out, "figure_s2_s3_training", title="Training out-of-fold predictions", dca_range=(.001, .999))
    comparison, score_predictions, score_mapping = compare_scores(train, test, result.predictions["test"]["LGBM"], heldout_thresholds["LGBM"])
    comparison.to_csv(out / "table_3_test_comparison.csv", index=False)
    score_mapping.to_csv(out / "score_probability_mapping.csv", index=False)
    raw_scores = {"LGBM": result.predictions["test"]["LGBM"], **{label: test[col].to_numpy() for label, col in SCORE_COLUMNS.items()}}
    plot_curves(test[TARGET], score_predictions, out, "figure_3_test", roc_scores=raw_scores, title="Prospective test set")
    subgroup_analysis(test, result.predictions["test"]["LGBM"], out, n_bootstraps=config["subgroup_bootstraps"], seed=config["subgroup_seed"])
    if not args.no_shap:
        explain_lgbm(result.models["LGBM"], test, out, allow_individual_plots=True)
    write_json(out / "run_manifest.json", {"status": "complete", "mode": args.mode, "profile": args.profile,
        "synthetic": args.mode == "synthetic", "configuration": config, "environment": environment(),
        "elapsed_seconds": round(time.monotonic() - start, 2), "selected_model": best_model,
        "feature_selection_skipped": args.skip_feature_selection, "shap_skipped": args.no_shap,
        "individual_record_csv_written": False, "individual_mark_figures_written": not args.no_shap,
        "model_weights_written": False})
    print("Analysis complete. The run manifest records the input mode and settings. Synthetic runs demonstrate the workflow.", flush=True)
    return 0
