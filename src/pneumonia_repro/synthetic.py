"""Generate independent demonstration cohorts entirely from random distributions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .data import CANDIDATE_COLS, SCORE_COLS, HISTORICAL_EXCLUDED, TARGET, calculate_egfr


def _cohort(n: int, rng: np.random.Generator, missing_rate: float, shift: float, missing_scores: bool = False) -> pd.DataFrame:
    if n < 50:
        raise ValueError("Use at least 50 rows per cohort to exercise stratified model training.")
    age = np.clip(rng.normal(78 + shift, 7.5, n), 65, 102).round()
    sex = rng.choice([1, 2], n, p=[0.58, 0.42])
    burden = rng.normal(size=n)
    immune = rng.binomial(1, 0.16, n)
    adl = np.clip(84 - 18 * burden + rng.normal(0, 13, n), 0, 100).round()
    neu = np.exp(rng.normal(1.65 + 0.22 * burden, 0.43, n))
    lym = np.exp(rng.normal(0.1 - 0.15 * burden, 0.42, n))
    hb = np.clip(rng.normal(126 - 4 * burden, 16, n), 65, 180)
    hypoxia = rng.binomial(1, 1 / (1 + np.exp(1.3 - 0.55 * burden)), n)
    cr = np.clip(np.exp(rng.normal(4.35 + 0.10 * burden, 0.35, n)), 25, 450)
    frame = pd.DataFrame({
        "age": age, "sex": sex, "height": rng.normal(164, 8, n).clip(140, 190),
        "weight": rng.normal(63, 12, n).clip(35, 110), "ADL_admission": adl,
        "Neu": neu, "Lym": lym, "O_PSI": hypoxia, "immunecompromised": immune,
        "Hb": hb, "Cr": cr,
        "smoking": rng.choice(["0 从未吸烟", "1 目前吸烟", "2 已戒烟"], n, p=[0.6, 0.2, 0.2]),
        "RR": rng.normal(22 + 1.5 * burden, 4, n).clip(12, 45).round(),
        "HR": rng.normal(84 + 5 * burden, 15, n).clip(40, 160).round(),
        "SBP": rng.normal(126 - 2 * burden, 19, n).clip(70, 200).round(),
        "DBP": rng.normal(72, 11, n).clip(35, 115).round(),
        "Mono": np.exp(rng.normal(-0.6, 0.4, n)), "CV": rng.normal(30, 3, n).clip(20, 45),
        "CRP": np.exp(rng.normal(3.7 + 0.2 * burden, 0.8, n)),
        "Urea": np.exp(rng.normal(1.7 + 0.15 * burden, 0.4, n)),
        "ALB": rng.normal(36 - burden, 4, n).clip(18, 50),
        "Glu": np.exp(rng.normal(1.8, 0.25, n)), "K": rng.normal(4.1, 0.4, n).clip(2.7, 6.0),
        "Na": rng.normal(138, 4, n).clip(120, 155), "TBIL": np.exp(rng.normal(2.4, 0.4, n)),
        "DBIL": np.exp(rng.normal(1.1, 0.4, n)), "ALT": np.exp(rng.normal(3.1, 0.5, n)),
        "AST": np.exp(rng.normal(3.2, 0.5, n)), "PT": rng.normal(13, 1.2, n).clip(9, 20),
        "APTT": rng.normal(30, 4, n).clip(18, 60), "Ddimer": np.exp(rng.normal(0.0, 0.8, n)),
        "hsTNI": np.exp(rng.normal(-2, 0.8, n)),
    })
    frame["WBC"] = frame["Neu"] + frame["Lym"] + frame["Mono"] + rng.uniform(0.1, 0.6, n)
    frame["BMI"] = frame["weight"] / frame["height"]**2 * 10000
    frame["eGFR_new"] = calculate_egfr(frame)
    rates = {"COPD": 0.2, "hypertension": 0.5, "DM": 0.25, "CAD": 0.2,
             "cerobrovascular": 0.2, "liver_disease": 0.06, "confusion": 0.12, "fever": 0.5,
             "CKD": 0.12, "tumor": 0.12, "ICU": 0.08, "septicshock": 0.05,
             "HFNC": 0.12, "NIV": 0.06, "IMV": 0.03, "nasal_mask": 0.55}
    for col, rate in rates.items():
        frame[col] = rng.binomial(1, rate, n)
    frame["respiratory_disease"] = np.maximum(frame["COPD"], rng.binomial(1, 0.15, n))
    frame["immunecompromised_type"] = np.where(immune, rng.integers(1, 5, n), 0)
    frame["Oxygen"] = rng.binomial(1, 0.2, n)
    frame["hospital_stay"] = np.exp(rng.normal(2.3, 0.4, n)).round().clip(2, 80)
    frame["imaging"] = rng.choice(["1 单侧", "2 双侧"], n, p=[0.45, 0.55])
    frame["pleural_effusion"] = rng.choice(["0 无", "1 单侧", "2 双侧"], n, p=[0.8, 0.1, 0.1])
    risk_logit = -2.8 + 0.035 * (75 - adl) + 0.09 * (neu - 6) - 0.012 * (hb - 120) + 0.7 * hypoxia + 0.6 * immune - 0.4 * (lym - 1)
    outcome = rng.binomial(1, 1 / (1 + np.exp(-risk_logit)))
    # Ensure enough events for five-fold cross-validation in small demonstration cohorts.
    if outcome.sum() < 10:
        candidates = np.flatnonzero(outcome == 0)
        outcome[rng.choice(candidates, 10 - int(outcome.sum()), replace=False)] = 1
    if outcome.sum() > n - 10:
        candidates = np.flatnonzero(outcome == 1)
        outcome[rng.choice(candidates, int(outcome.sum()) - (n - 10), replace=False)] = 0
    frame[TARGET] = outcome
    frame["discharge"] = np.where((outcome == 1) & (rng.uniform(size=n) < 0.65), "3 死亡", "1 好转")
    curb = (age >= 65).astype(int) + (frame["Urea"] > 7).astype(int) + (frame["RR"] >= 30).astype(int) + ((frame["SBP"] < 90) | (frame["DBP"] <= 60)).astype(int) + frame["confusion"]
    frame["CURB_65"] = curb
    frame["CRB_65"] = curb - (frame["Urea"] > 7).astype(int)
    frame["PSI"] = np.clip(age + 15 * burden + 20 * hypoxia + rng.normal(0, 12, n), 30, 220).round()
    frame["SMART_COP"] = np.clip((2 * hypoxia + frame["confusion"] + (frame["ALB"] < 35).astype(int) + (frame["RR"] >= 30).astype(int) + rng.integers(0, 2, n)), 0, 8)
    frame["ADROP"] = np.clip((age >= 70).astype(int) + hypoxia + (frame["Urea"] > 7).astype(int) + frame["confusion"] + (frame["SBP"] < 90).astype(int), 0, 5)
    for col in HISTORICAL_EXCLUDED:
        frame[col] = np.exp(rng.normal(3, 0.6, n))
        frame.loc[rng.uniform(size=n) < 0.65, col] = np.nan
    for col in CANDIDATE_COLS:
        if col not in HISTORICAL_EXCLUDED + ["age", "sex"] and col in frame:
            frame.loc[rng.uniform(size=n) < missing_rate, col] = np.nan
    # Optional missing scores remain missing after preprocessing to exercise
    # the score comparison's explicit rejection of missing scores.
    if missing_scores:
        for col in SCORE_COLS:
            frame.loc[rng.uniform(size=n) < min(missing_rate / 3, 0.02), col] = np.nan
    return frame


def generate_cohorts(
    n_retrospective: int = 822, n_prospective: int = 545,
    seed: int = 42, missing_rate: float = 0.03, missing_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return random cohorts with standardized fields and complete outcomes before missingness screening."""
    if not 0 <= missing_rate <= 0.15:
        raise ValueError("missing_rate must be between 0 and 0.15 for a usable demonstration.")
    rng = np.random.default_rng(seed)
    return (_cohort(n_retrospective, rng, missing_rate, 0, missing_scores),
            _cohort(n_prospective, rng, missing_rate, -1, missing_scores))


def write_synthetic_data(
    output_dir: str | Path, n_retrospective: int = 822, n_prospective: int = 545,
    seed: int = 42, missing_rate: float = 0.03, missing_scores: bool = False,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ret, pro = generate_cohorts(n_retrospective, n_prospective, seed, missing_rate, missing_scores)
    paths = {"retrospective": out / "retrospective.csv", "prospective": out / "prospective.csv"}
    ret.to_csv(paths["retrospective"], index=False)
    pro.to_csv(paths["prospective"], index=False)
    (out / "SYNTHETIC_DATA.json").write_text(json.dumps({
        "synthetic": True, "patient_records_used": False, "seed": seed,
        "n_retrospective": n_retrospective, "n_prospective": n_prospective,
        "missing_rate": missing_rate, "outcome_encoding": "1=death;0=survival",
        "missing_scores": missing_scores,
        "purpose": "Demonstrate the analysis workflow using synthetic data.",
        "scores": "Simulated comparator scores.",
    }, indent=2) + "\n", encoding="utf-8")
    return paths
