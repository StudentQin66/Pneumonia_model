"""Reconstruct the 87 rows in the supplementary missingness tables.

Date fields are reduced to counts of nonmissing values immediately after
reading. The tables are computed independently of feature screening and imputation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .data import RAW_COLUMN_MAP, calculate_egfr

# List the source fields in manuscript order. Both eosinophil labels refer
# to the same source field, as noted in the output table.
MISSINGNESS_FIELDS = [
    ("age", "age"), ("sex", "sex"), ("smoking", "smoking history"),
    ("weight", "weight"), ("height", "height"),
    ("ADL_admission", "activities of daily living score on admission"),
    ("ADL_discharge", "activities of daily living score on discharge"),
    ("COPD", "chronic obstructive pulmonary disease"), ("bronchiectasis", "bronchiectasis"),
    ("ILD", "interstitial lung disease"), ("PH", "pulmonary hypertension"),
    ("asthma", "asthma"), ("hypertension", "hypertension"), ("CAD", "coronary heart disease"),
    ("MI", "myocardial infarction"), ("HF", "congestive heart failure"),
    ("virus_hepatisis", "viral hepatitis"), ("alcholic_hepatopathy", "alcoholic hepatitis"),
    ("antoimmune_hepatopathy", "autoimmune liver disease"), ("cirrhosis", "cirrhosis"),
    ("PBC", "primary biliary cholangitis"),
    ("chronic_renal_insufficiency", "chronic kidney insufficiency"),
    ("Nephrotic_syndrome", "nephrotic syndrome"), ("diabetic_nephropathy", "diabetic nephropathy"),
    ("Glomerulonephritis", "glomerulonephritis"), ("Hypertensive_nephropathy", "hypertensive nephropathy"),
    ("cerobrovascular", "cerebral infarction"), ("DM", "diabetes"), ("tumor", "malignant tumor"),
    ("immunecompromised", "immune function"), ("fever", "fever"), ("confusion", "confusion"),
    ("HR", "heart rate"), ("RR", "respiratory rate"), ("SBP", "systolic blood pressure"),
    ("DBP", "diastolic blood pressure"), ("SpO2_actual", "oxygen saturation of the pulse"),
    ("WBC", "white blood cell"), ("Hb", "hemoglobin"), ("HCT", "hematocrit"),
    ("CV", "mean corpuscular volume"), ("Eos", "percentage of eosinophilic cells"),
    ("Neu", "absolute neutrophil count"), ("Lym", "absolute lymphocytes count"),
    ("Mono", "absolute monocyte count"), ("Eos", "absolute eosinophilic count"),
    ("Glu", "blood glucose"), ("ALB", "albumin"), ("Cr", "creatinine"),
    ("eGFR_new", "estimated glomerular filtration rate"), ("CK", "creatine kinase"),
    ("LDH", "lactic dehydrogenase"), ("K", "blood potassium"), ("Na", "blood sodium"),
    ("Urea", "urea"), ("TBIL", "total bilirubin"), ("DBIL", "direct bilirubin"),
    ("ALT", "alanine aminotransferase"), ("AST", "aspartate aminotransferase"),
    ("ferritin", "serum ferritin"), ("PT", "prothrombin time"),
    ("APTT", "activated partial thromboplastin time"), ("Ddimer", "D-dimer"),
    ("hsTNT", "high-Sensitivity cardiac troponin T"), ("hsTNI", "high-Sensitivity cardiac troponin I"),
    ("BNP", "BNP"), ("NTproBNP", "NT-proBNP"), ("FIO2", "FiO2"),
    ("PH_A", "PH value"), ("PaO2", "PaO2"), ("PaCO2", "PaCO2"), ("Lac", "lactic acid"),
    ("pleural_effusion", "pleural effusion"), ("imaging", "lung filtration"),
    ("admission_date", "the day of admission"), ("discharge_date", "the day of discharge"),
    ("ICU", "intensive care unit admission"), ("ICU_admissiondate", "the day of ICU admission"),
    ("Oxygen", "Receive oxygen therapy"), ("nasal_mask", "nasal cannula or oxygen mask"),
    ("HFNC", "high flow nasal cannula"), ("NIV", "noninvasive mechanical ventilation"),
    ("IMV", "invasive mechanical ventilation"), ("septicshock", "septic shock"),
    ("hospital_stay", "length of hospital stay"), ("day_30mortality", "30-day mortality"),
    ("death", "In-hospital mortality"),
]

EXTRA_RAW_MAP = {
    "肺动脉高压": "PH", "病毒性肝炎": "virus_hepatisis", "酒精性肝病": "alcholic_hepatopathy",
    "自身免疫性肝病": "antoimmune_hepatopathy", "肝硬化": "cirrhosis", "原发性胆汁性肝硬化": "PBC",
    "慢性肾功能不全": "chronic_renal_insufficiency", "肾病综合征": "Nephrotic_syndrome",
    "糖尿病肾病": "diabetic_nephropathy", "肾小球肾炎": "Glomerulonephritis", "高血压肾病": "Hypertensive_nephropathy",
    "实测SpO2": "SpO2_actual", "实测FI02": "FIO2", "血气Lac": "Lac",
    "入院日期": "admission_date", "出院日期": "discharge_date", "入住监护室时间": "ICU_admissiondate",
}
DATE_FIELDS = {"admission_date", "discharge_date", "ICU_admissiondate"}


def manuscript_missingness_table(
    source: pd.DataFrame | str | Path,
    *, canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return aggregate counts for the 87 rows in manuscript order.

    If supplied, the unfiltered canonical cohort provides the eGFR, height,
    and weight values already prepared for this run. Otherwise, use eGFR_new
    from the source or calculate it using the 2019 formula for the Chinese
    population. Fields absent from the input are marked as unavailable.
    """
    source_fields = {f for f, _ in MISSINGNESS_FIELDS}
    source_fields.update({"discharge", "Cr", "age", "sex", "eGFR_new"})
    mapping = {k: v for k, v in {**RAW_COLUMN_MAP, **EXTRA_RAW_MAP}.items() if v in source_fields}
    allowed = source_fields | set(mapping)
    # Distinguish the two meanings of PH using the source headers. PH denotes
    # blood gas pH in the Chinese prospective file and pulmonary hypertension
    # in the English retrospective export.
    allowed.add("PH")
    if isinstance(source, pd.DataFrame):
        frame = source.loc[:, source.columns.isin(allowed)].copy()
    else:
        path = Path(source)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, usecols=lambda c: c in allowed)
        else:
            frame = pd.read_csv(path, usecols=lambda c: c in allowed)
    if "性别" in frame:
        mapping["PH"] = "PH_A"
    frame = frame.rename(columns={c: mapping[c] for c in frame if c in mapping})
    if frame.columns.duplicated().any():
        raise ValueError("Multiple input columns map to the same missingness field. Provide one column per field.")
    n = len(frame)
    if canonical is not None and len(canonical) != n:
        raise ValueError("Missingness requires the unfiltered canonical cohort, not an eligible subset.")
    # Count observed dates and discard the original values.
    date_counts = {c: int(frame[c].notna().sum()) for c in DATE_FIELDS if c in frame}
    frame = frame.drop(columns=list(DATE_FIELDS), errors="ignore")
    if canonical is not None:
        for field in ["eGFR_new", "height", "weight", "fever", "death"]:
            if field in canonical:
                frame[field] = canonical[field].to_numpy()
    if "eGFR_new" not in frame and {"Cr", "age", "sex"}.issubset(frame):
        frame["eGFR_new"] = calculate_egfr(frame)
    if "death" not in frame and "discharge" in frame:
        # Use discharge status to count records with an observed outcome.
        frame["death"] = frame["discharge"].notna().where(frame["discharge"].notna())
    for field in ["height", "weight"]:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    counts = {c: int(frame[c].notna().sum()) for c in source_fields if c in frame}
    counts.update(date_counts)
    del frame
    rows = []
    for order, (field, label) in enumerate(MISSINGNESS_FIELDS, start=1):
        available = field in counts
        observed = counts.get(field, np.nan)
        notes = []
        if field in DATE_FIELDS:
            notes.append("Aggregate source availability only; date values discarded.")
        if field == "SpO2_actual":
            notes.append("Missing count is calculated as N minus the observed count.")
        if field == "eGFR_new":
            notes.append("Analysis-derived eGFR_new, not the different source eGFR field.")
        if field == "CV":
            notes.append("Uses the source field CV.")
        if field == "Eos":
            notes.append("Both eosinophil rows use the source field Eos.")
        rows.append({"Manuscript_order": order, "Variable": field, "Label": label,
                     "N": n, "Observed": observed,
                     "Missing": n - observed if available else np.nan,
                     "Missing_percent": 100 * (n - observed) / n if available and n else np.nan,
                     "Status": "available" if available else "unavailable",
                     "Note": " ".join(notes)})
    return pd.DataFrame(rows)
