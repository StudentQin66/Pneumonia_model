#!/usr/bin/env python3
"""Run the manuscript analysis from the repository source tree."""
from pathlib import Path
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_repro.run import main

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message=(
            r"^X does not have valid feature names, but LGBMClassifier "
            r"was fitted with feature names$"
        ),
        category=UserWarning,
        module=r"^sklearn\.utils\.validation$",
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"^LightGBM binary classifier with TreeExplainer shap values output "
            r"has changed to a list of ndarray$"
        ),
        category=UserWarning,
        module=r"^shap\.explainers\._tree$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^The max_iter was reached which means the coef_ did not converge$",
        category=ConvergenceWarning,
        module=r"^sklearn\.linear_model\._sag$",
    )
    raise SystemExit(main())
