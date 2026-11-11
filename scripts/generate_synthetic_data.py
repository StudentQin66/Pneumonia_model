#!/usr/bin/env python3
"""Generate synthetic CSV files independently of the study data using the analysis schema."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_repro.synthetic import write_synthetic_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("--output-dir", default="outputs/synthetic-data", help="Directory for the synthetic CSV files and generation record (default outputs/synthetic-data).")
    parser.add_argument("--n-retrospective", type=int, default=822, help="Number of synthetic retrospective records (default 822).")
    parser.add_argument("--n-prospective", type=int, default=545, help="Number of synthetic prospective records (default 545).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--missing-rate", type=float, default=0.03, help="Random missingness rate for candidate predictors (default 0.03).")
    parser.add_argument("--missing-scores", action="store_true", help="Generate missing scores to check the comparison routine's input validation. The standard demo uses complete scores.")
    args = parser.parse_args()
    paths = write_synthetic_data(args.output_dir, args.n_retrospective, args.n_prospective, args.seed, args.missing_rate, args.missing_scores)
    print("Synthetic data generated for the workflow demonstration.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
