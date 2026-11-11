# Reproducibility code for 30-day mortality prediction in older adults with community-acquired pneumonia

This repository provides the analysis code and instructions for the manuscript *Development and validation of an interpretable machine learning model for predicting mortality in elderly community-acquired pneumonia patients*. The analysis starts with the retrospective and prospective cohorts after clinical enrollment and generates the results for main-text Tables 1–3 and Figures 1–5, as well as Supplementary Tables S1–S8 and Figures S1–S3. Reviewers can run the main workflow using independently generated synthetic data.

| File or directory | Purpose |
| --- | --- |
| [ANALYSIS_MAP.md](ANALYSIS_MAP.md) | Maps each manuscript table and figure to its analysis functions and output files |
| [requirements.txt](requirements.txt) and [requirements-lock.txt](requirements-lock.txt) | List the main analysis dependencies and the full set of Python package versions used in the validated environment |
| [config/](config/) | Contains the quick demonstration settings in `smoke.json` and the manuscript settings in `paper.json` |
| [scripts/](scripts/) | Contains the analysis entry point `run_analysis.py` and the standalone synthetic data generator `generate_synthetic_data.py` |
| [src/pneumonia_repro/](src/pneumonia_repro/) | Contains the code for data processing, missingness summaries, tables, modeling, evaluation, plotting, and synthetic data generation |

## Installation

Run the following commands from the repository root.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.13.5 to create the virtual environment. To install the full set of Python package versions recorded in the validated environment, replace the last line with `python -m pip install -r requirements-lock.txt`. 

The `.venv/` and `outputs/` directories are generated locally. The release contains the source code, configuration, and documentation.

## Quick run on synthetic data

```bash
python -B scripts/run_analysis.py --mode synthetic --profile smoke \
  --output-dir outputs/smoke
```

The program generates synthetic retrospective and prospective cohorts of 360 and 240 records in memory. It screens for missing data, splits the development cohort, imputes missing values, selects features, trains and calibrates models using seven algorithms, and runs clinical score comparisons, subgroup analyses, and SHAP analyses. Use a new or empty output directory for each run. With the dependencies already installed, this example takes about 29 seconds in the validated environment.

The `smoke` profile uses fewer cross-validation folds, hyperparameter search iterations, and bootstrap replicates. The next command uses the full manuscript settings with the same synthetic cohort sizes and takes about 81 seconds in the validated environment.

```bash
python -B scripts/run_analysis.py --mode synthetic --profile paper \
  --output-dir outputs/synthetic-paper
```

The program randomly generates synthetic variables, missing values, and comparator scores according to predefined rules. Model performance metrics vary with the synthetic data and serve to illustrate the workflow.

## Export synthetic data

The following command exports synthetic CSV files so that you can inspect the input fields and format.

```bash
python -B scripts/generate_synthetic_data.py --n-retrospective 100 \
  --n-prospective 80 --seed 42 \
  --output-dir outputs/synthetic-data
```

The output includes `retrospective.csv`, `prospective.csv`, and `SYNTHETIC_DATA.json`, which records the generation settings. You can adjust the cohort sizes and random seed. The quick run generates its own inputs.

## Run with authorized original data

If you have authorized access to the original data, use the two original Excel files containing all enrolled participants. Preserve the original row order and replace the input paths in the command with the paths to your files. The study encodes 30-day mortality after admission as `1=death` and `0=survival` and uses the equation for the Chinese population to calculate `eGFR_new`.

```bash
python -B scripts/run_analysis.py --mode real --profile paper \
  --retrospective ../controlled-data/retrospective.xlsx \
  --prospective ../controlled-data/prospective.xlsx \
  --prospective-outcome death --egfr-formula china2019 \
  --missingness-policy manuscript \
  --output-dir outputs/real-paper
```

## View results

Results are saved to the directory specified by `--output-dir`. When the run finishes, `run_manifest.json` records the run settings and software versions, with `status` set to `complete`. Open `table_2_validation.csv` for the internal validation results and `table_3_test_comparison.csv` for the comparison of the model with clinical scores in the prospective test cohort. The remaining tables, PNG and PDF figures, and their underlying data are listed in the [analysis map](ANALYSIS_MAP.md).
