"""Define the input schema and preprocessing fitted on the training set.

The reader loads the allowed clinical fields. The default missingness rules
use the candidate predictors specified in the manuscript. An explicit option
reproduces the original notebook's missingness rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

TARGET = "day_30mortality"
SCORE_COLS = ["CURB_65", "CRB_65", "PSI", "SMART_COP", "ADROP"]
FINAL_FEATURES = ["ADL_admission", "Neu", "O_PSI", "immunecompromised", "Hb", "eGFR_new", "Lym"]
CONTINUOUS_COLS = [
    "age", "sex", "BMI", "RR", "HR", "SBP", "DBP", "WBC", "Neu", "Lym",
    "Mono", "Hb", "CV", "CRP", "Urea", "ALB", "Glu", "eGFR_new", "K", "Na",
    "TBIL", "DBIL", "ALT", "AST", "PT", "APTT", "Ddimer", "hsTNI", "ADL_admission",
]
BINARY_COLS = [
    "Never_smoker", "Current_smoker", "Former_smoker", "COPD", "hypertension",
    "DM", "CAD", "cerobrovascular", "liver_disease", "immunecompromised",
    "confusion", "fever", "O_PSI",
]
# Keep the order of unique fields in select_var.xlsx, including the six
# variables later excluded for high missingness. Clinical scores are supplied separately.
CANDIDATE_COLS = [
    "age", "sex", "BMI", "smoking", "COPD", "hypertension", "DM", "CAD",
    "cerobrovascular", "liver_disease", "immunecompromised", "confusion", "fever",
    "RR", "HR", "SBP", "DBP", "O_PSI", "WBC", "Neu", "Lym", "Mono", "Hb",
    "CV", "CRP", "Urea", "ALB", "Glu", "eGFR_new", "CK", "LDH", "K", "Na",
    "TBIL", "DBIL", "ALT", "AST", "ferritin", "PT", "APTT", "Ddimer", "hsTNT",
    "hsTNI", "BNP", "NTproBNP", "ADL_admission",
]
HISTORICAL_EXCLUDED = ["CK", "LDH", "ferritin", "hsTNT", "BNP", "NTproBNP"]

# Map the source clinical fields used for modeling and descriptive analyses.
RAW_COLUMN_MAP = {
    "性别": "sex", "年龄": "age", "吸烟情况": "smoking", "住院时间": "hospital_stay",
    "是否入住监护室": "ICU", "住院期间未接受额外氧疗": "Oxygen",
    "住院期间接受鼻导管或面罩氧疗": "nasal_mask", "住院期间接受经鼻高流量氧疗": "HFNC",
    "是否接受无创机械通气": "NIV", "是否接受有创机械通气": "IMV",
    "出院时状态": "discharge", "入院30天内生存": TARGET, "是否出现感染性休克": "septicshock",
    "是否需要血管活性药物": "vasoactive_drug", "入院ADL": "ADL_admission", "出院ADL": "ADL_discharge",
    "呼吸系统疾病": "respiratory_disease", "慢性阻塞性肺疾病": "COPD", "支气管扩张": "bronchiectasis",
    "间质性肺疾病": "ILD", "支气管哮喘": "asthma", "心血管系统疾病": "CVD",
    "高血压": "hypertension", "冠心病": "CAD", "心肌梗塞": "MI", "充血性心力衰竭": "HF",
    "肝脏疾病": "liver_disease", "慢性肾脏疾病": "CKD", "糖尿病": "DM", "脑血管疾病": "cerobrovascular",
    "恶性肿瘤": "tumor", "是否免疫抑制人群": "immunecompromised", "免疫抑制人群类型": "immunecompromised_type",
    "身高cm": "height", "体重Kg": "weight", "意识障碍": "confusion", "是否发热": "fever",
    "心率": "HR", "呼吸频率": "RR", "收缩压": "SBP", "舒张压": "DBP",
    "HGB": "Hb", "Urea_mmol每升": "Urea", "BUN_mg每分升": "BUN_mgdl", "铁蛋白": "ferritin",
    "胸腔积液": "pleural_effusion", "影像学肺脏炎症": "imaging",
    "新冠肺炎": "COVID_pneumonia", "流感病毒肺炎": "influenza_pneumonia", "真菌性肺炎": "fungal_pneumonia",
    "CURB65": "CURB_65", "CRB65": "CRB_65", "SMARTCOP": "SMART_COP",
}
BASELINE_EXTRA_COLS = [
    "height", "weight", "Cr", "respiratory_disease", "CKD", "tumor", "immunecompromised_type",
    "Oxygen", "nasal_mask", "HFNC", "NIV", "IMV", "septicshock", "hospital_stay", "ICU",
    "discharge", "discharge_death", "imaging", "pleural_effusion", "death", "bi", "ui", "bpe", "upe",
    "ADL_discharge", "vasoactive_drug", "COVID_pneumonia", "influenza_pneumonia", "fungal_pneumonia",
    "bronchiectasis", "ILD", "asthma", "CVD", "MI", "HF", "BUN_mgdl", "eGFR",
]
ALLOWED_INPUTS = set(CANDIDATE_COLS + CONTINUOUS_COLS + BINARY_COLS + SCORE_COLS + BASELINE_EXTRA_COLS + [TARGET])
ALLOWED_INPUTS.update(RAW_COLUMN_MAP)


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    retrospective: pd.DataFrame
    prospective: pd.DataFrame
    audit: dict[str, Any]
    preprocessor: ColumnTransformer


def _read_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.loc[:, source.columns.isin(ALLOWED_INPUTS)].copy()
    else:
        path = Path(source)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, usecols=lambda c: c in ALLOWED_INPUTS)
        else:
            frame = pd.read_csv(path, usecols=lambda c: c in ALLOWED_INPUTS)
    if frame.empty:
        raise ValueError("Input has no participant rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names.")
    return frame


def _numeric(series: pd.Series, column: str) -> pd.Series:
    # Accept SPSS-style value labels with numeric prefixes.
    text = series.astype("string").str.strip()
    text = text.replace({"": pd.NA, ".": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA})
    result = pd.to_numeric(text, errors="coerce")
    labeled = text.str.extract(r"^(-?\d+(?:\.\d+)?)\s+[^\d]", expand=False)
    result = result.fillna(pd.to_numeric(labeled, errors="coerce"))
    bad = text.notna() & result.isna()
    if bad.any():
        raise ValueError(f"Column {column} has {int(bad.sum())} nonmissing values that could not be converted to numbers.")
    result = result.astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError(f"Column {column} contains infinite values.")
    return result


def _smoking_codes(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA, "NA": pd.NA})
    result = pd.to_numeric(text, errors="coerce")
    # The original prospective workbook includes the label for current smokers
    # with and without a space after 1. The corrected CSV standardizes the spacing.
    prefix = text.str.extract(r"^([012])(?:\s|[^\d.]|$)", expand=False)
    result = result.fillna(pd.to_numeric(prefix, errors="coerce"))
    if (text.notna() & result.isna()).any() or not result.dropna().isin([0, 1, 2]).all():
        raise ValueError("smoking must use 0=never, 1=current, 2=former (numeric or source labels).")
    return result.astype(float)


def calculate_egfr(frame: pd.DataFrame, formula: str = "china2019") -> pd.Series:
    """Apply the formulas from cells 2 and 3 of the original preprocessing notebook.

    Creatinine (Cr) is supplied in µmol/L.
    """
    if formula not in {"china2019", "ckdepi2021"}:
        raise ValueError("egfr_formula must be china2019, ckdepi2021, or source_final when eGFR_new is supplied.")
    for col in ("Cr", "age", "sex"):
        if col not in frame:
            raise ValueError(f"Deriving eGFR requires {col}.")
    scr = _numeric(frame["Cr"], "Cr") / 88.4
    age = _numeric(frame["age"], "age")
    sex = _numeric(frame["sex"], "sex")
    female = sex.isin([0, 2])
    if ((scr <= 0) & scr.notna()).any():
        raise ValueError("Cr must be positive when eGFR is calculated.")
    k = np.where(female, 0.7, 0.9)
    ratio = scr / k
    if formula == "china2019":
        coefficient = np.where(female, 130.0, 135.0)
        exponent = np.where(female, np.where(ratio <= 1, -0.248, -1.098), np.where(ratio <= 1, -0.207, -1.194))
        result = coefficient * ratio**exponent * 0.994**age
    else:
        alpha = np.where(female, -0.241, -0.302)
        result = 142 * np.minimum(ratio, 1)**alpha * np.maximum(ratio, 1)**(-1.2) * 0.9938**age * np.where(female, 1.012, 1)
    return pd.Series(result, index=frame.index).where(sex.isin([0, 1, 2]))


def canonicalize(
    source: pd.DataFrame | str | Path, *, prospective: bool = False,
    prospective_outcome: str | None = None, egfr_formula: str | None = None,
) -> pd.DataFrame:
    """Read the allowed clinical fields and check the outcome coding.

    The canonical field day_30mortality and the original prospective field
    labeled "30-day survival after admission" both use 1 for death and 0 for survival.
    """
    frame = _read_frame(source)
    raw_survival_header = "入院30天内生存" in frame
    rename = {c: RAW_COLUMN_MAP[c] for c in frame if c in RAW_COLUMN_MAP}
    if len(set(rename.get(c, c) for c in frame)) != len(frame.columns):
        raise ValueError("Multiple input columns map to the same analysis field. Provide one column per field.")
    frame = frame.rename(columns=rename)
    if TARGET not in frame:
        raise ValueError(f"Input is missing required outcome {TARGET}.")
    if raw_survival_header and prospective and prospective_outcome is None:
        # The original prospective outcome uses 1 for death and 0 for survival.
        prospective_outcome = "death"
    if prospective_outcome not in {None, "death", "survival"}:
        raise ValueError("prospective_outcome must be death or survival.")
    # Retain source values for the six fields with high missingness to reproduce
    # the original screening. Strings such as detection limits count as observed.
    text_fields = {"smoking", "discharge", "imaging", "pleural_effusion", *HISTORICAL_EXCLUDED}
    for col in frame.columns.difference(list(text_fields)):
        if col in {"height", "weight"}:
            # Use the numeric conversion from cells 4 and 5 of the preprocessing notebook.
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = _numeric(frame[col], col)
    outcome = frame[TARGET]
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError("All outcomes must be observed binary 0/1; missing outcomes are never imputed.")
    # Apply explicit survival coding to prospective inputs, including exports
    # whose columns already use the canonical names.
    if prospective and prospective_outcome == "survival":
        frame[TARGET] = 1 - outcome
    frame[TARGET] = frame[TARGET].astype(int)
    if "sex" not in frame or not frame["sex"].dropna().isin([0, 1, 2]).all():
        raise ValueError("sex must use 1=male, 2 or 0=female.")
    if "BMI" not in frame:
        if not {"height", "weight"}.issubset(frame):
            raise ValueError("BMI or both height (cm) and weight (kg) are required.")
        frame["BMI"] = frame["weight"] / frame["height"]**2 * 10000
    if "eGFR_new" not in frame:
        if egfr_formula == "source_final":
            raise ValueError("egfr_formula=source_final requires supplied eGFR_new.")
        frame["eGFR_new"] = calculate_egfr(frame, egfr_formula or "china2019")
    elif egfr_formula not in {None, "source_final"}:
        frame["eGFR_new"] = calculate_egfr(frame, egfr_formula)
    frame["sex"] = frame["sex"].replace(2, 0)
    if "smoking" in frame:
        frame["smoking"] = _smoking_codes(frame["smoking"]).map({0: "0 从未吸烟", 1: "1 目前吸烟", 2: "2 已戒烟"})
    for col in set(BINARY_COLS) - {"Never_smoker", "Current_smoker", "Former_smoker", "fever"}:
        if col in frame and not frame[col].dropna().isin([0, 1]).all():
            raise ValueError(f"Binary column {col} must use 0/1.")
    # Derive baseline indicators by matching the original category codes.
    for out, col, code in [("death", "discharge", 3), ("bi", "imaging", 2), ("ui", "imaging", 1), ("bpe", "pleural_effusion", 2), ("upe", "pleural_effusion", 1)]:
        if col in frame:
            number = pd.to_numeric(frame[col].astype("string").str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")
            frame[out] = number.eq(code).fillna(False).astype(int)
    return frame


def _smoking_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "smoking" in frame:
        smoking = _smoking_codes(frame["smoking"])
        if not smoking.dropna().isin([0, 1, 2]).all():
            raise ValueError("smoking must use 0=never, 1=current, 2=former.")
        # The original notebook sets all three indicators to 0 for missing smoking status.
        for code, col in enumerate(["Never_smoker", "Current_smoker", "Former_smoker"]):
            frame[col] = smoking.eq(code).astype(int)
        frame = frame.drop(columns="smoking")
    elif not {"Never_smoker", "Current_smoker", "Former_smoker"}.issubset(frame):
        raise ValueError("smoking or all three smoking indicators are required.")
    return frame


def prepare_data(
    retrospective: pd.DataFrame | str | Path,
    prospective: pd.DataFrame | str | Path,
    *, prospective_outcome: str | None = None, egfr_formula: str | None = None,
    missingness_policy: str = "manuscript", seed: int = 42,
) -> PreparedData:
    """Screen for missingness, split the development cohort 80:20, and fit imputation on training data.

    The legacy-complete-identifiers option reproduces the original notebook's
    denominator by counting one additional, fully observed identifier field.
    The calculation uses clinical fields and adjusts the denominator accordingly.
    """
    if missingness_policy not in {"manuscript", "legacy-complete-identifiers"}:
        raise ValueError("Unknown missingness_policy.")
    raw_ret = canonicalize(retrospective, egfr_formula=egfr_formula)
    raw_pro = canonicalize(prospective, prospective=True, prospective_outcome=prospective_outcome, egfr_formula=egfr_formula)
    required = set(CONTINUOUS_COLS + [c for c in BINARY_COLS if not c.endswith("_smoker")] + SCORE_COLS + [TARGET])
    for label, frame in [("retrospective", raw_ret), ("prospective", raw_pro)]:
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} input lacks required analysis fields: {', '.join(missing)}")
    # Check that both cohorts contain the variables used for screening.
    candidates = [c for c in CANDIDATE_COLS if c in raw_ret]
    if "smoking" not in candidates:
        candidates += [c for c in ["Never_smoker", "Current_smoker", "Former_smoker"] if c in raw_ret]
    missing_pro = sorted(set(candidates) - set(raw_pro))
    if missing_pro:
        raise ValueError("Prospective input lacks retrospective candidate fields: " + ", ".join(missing_pro))
    eligibility_cols = candidates + (SCORE_COLS if missingness_policy != "manuscript" else [])
    high_missing = [c for c in eligibility_cols if raw_ret[c].isna().mean() > 0.30]
    retained = [c for c in candidates if c not in high_missing]
    if set(FINAL_FEATURES) & set(high_missing):
        raise ValueError("A reported final predictor exceeds 30% retrospective missingness; the reported model cannot be reconstructed on this input.")
    ret_denominator_cols = [c for c in eligibility_cols if c not in high_missing]
    ret_keep = raw_ret[ret_denominator_cols].isna().mean(axis=1) <= 0.20
    if missingness_policy == "manuscript":
        pro_denominator_cols = retained
        pro_keep = raw_pro[pro_denominator_cols].isna().mean(axis=1) <= 0.20
    else:
        # Cell 7 of the original main0904 notebook excludes six variables and
        # the clinical scores, and counts a fully observed name field in the denominator.
        pro_denominator_cols = [c for c in candidates if c not in HISTORICAL_EXCLUDED]
        pro_keep = raw_pro[pro_denominator_cols].isna().sum(axis=1) / (len(pro_denominator_cols) + 1) <= 0.20
    ret = _smoking_indicators(raw_ret.loc[ret_keep].copy())
    pro = _smoking_indicators(raw_pro.loc[pro_keep].copy())
    train, val = train_test_split(ret, test_size=0.2, train_size=0.8, random_state=seed, stratify=ret[TARGET])
    cont = [c for c in CONTINUOUS_COLS if c not in high_missing]
    binary = [c for c in BINARY_COLS if c not in high_missing]
    if train[cont + binary].isna().all().any():
        raise ValueError("A retained predictor is entirely missing in training; its imputation cannot be fitted.")
    preprocessor = ColumnTransformer([
        ("cont", IterativeImputer(max_iter=10, random_state=seed), cont),
        ("cat", SimpleImputer(strategy="most_frequent"), binary),
    ], verbose_feature_names_out=False)
    preprocessor.fit(train)
    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(preprocessor.transform(frame), columns=cont + binary, index=frame.index)
        result[binary] = result[binary].round().astype(int)
        result[TARGET] = frame[TARGET].astype(int)
        for col in SCORE_COLS:
            result[col] = frame[col]
        return result
    train_imputed, val_imputed, test_imputed = transform(train), transform(val), transform(pro)
    audit = {
        "missingness_policy": missingness_policy,
        "seed": seed, "outcome": "1=30-day death; 0=survival",
        "prospective_source_outcome": prospective_outcome or "canonical-death",
        "egfr_derivation": egfr_formula or "source eGFR_new if supplied; otherwise china2019",
        "high_missing_columns": high_missing,
        "retained_candidate_columns": retained,
        "retrospective_row_missingness_denominator": len(ret_denominator_cols),
        "prospective_row_missingness_denominator": len(pro_denominator_cols) + int(missingness_policy != "manuscript"),
        "legacy_complete_identifier_assumption": missingness_policy != "manuscript",
        "counts": {"retrospective_raw": len(raw_ret), "prospective_raw": len(raw_pro),
                   "retrospective_eligible": len(ret), "prospective_eligible": len(pro),
                   "train": len(train), "validation": len(val), "test": len(pro)},
        "deaths": {"train": int(train[TARGET].sum()), "validation": int(val[TARGET].sum()), "test": int(pro[TARGET].sum())},
        "imputation": {"fit_partition": "training only", "continuous": cont, "mode": binary,
                       "outcome_imputed": False, "scores_imputed": False},
    }
    return PreparedData(train_imputed, val_imputed, test_imputed, raw_ret, raw_pro, audit, preprocessor)
