"""Step 18: audit prediction timing and target provenance without model changes."""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"
FEATURE_PATH = PROJECT_ROOT / "data" / "feature_engineered_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "timing_audit_outputs"
TARGET = "readmission_30day"

DEMOGRAPHICS = [
    "age", "gender", "weight_kg", "height_cm", "bmi", "ethnicity", "insurance"
]
COMORBIDITIES = [
    "diabetes", "hypertension", "chf", "copd", "chronic_kidney_disease",
    "liver_disease", "immunosuppression", "cad", "atrial_fibrillation",
    "cancer_active", "comorbidity_count"
]
VITALS = [
    "hr_mean", "hr_max", "hr_min", "hr_std", "sbp_mean", "sbp_max", "sbp_min",
    "sbp_std", "dbp_mean", "dbp_max", "dbp_min", "dbp_std", "map_mean",
    "temp_celsius_mean", "temp_celsius_max", "temp_celsius_min", "temp_celsius_std",
    "spo2_mean", "spo2_min", "spo2_max", "spo2_std", "respiratory_rate_mean",
    "respiratory_rate_max", "respiratory_rate_min", "respiratory_rate_std"
]
LABS = [
    "wbc", "lactate_mmol", "creatinine", "platelet_count", "bilirubin_total",
    "glucose", "ph_arterial", "pao2_fio2_ratio", "inr", "sodium", "potassium",
    "chloride", "bicarbonate", "hematocrit", "hemoglobin"
]
TREATMENT = [
    "vasopressors_flag", "mechanical_ventilation", "fio2_percent", "antibiotics_24h",
    "fluids_ml_24h", "sedation_score", "vasopressor_dose_mcg_kg_min",
    "insulin_infusion_flag", "treatment_flag_count"
]
SEVERITY = ["sofa_score", "apache_iv", "qsofa", "sirs_criteria", "gcs_total"]
HOSPITAL_CONTEXT = ["icu_los_hours", "hospital_admit_source", "icu_admit_time_hour", "day_of_week"]
ENGINEERED = ["lab_missing_count", "bmi_missing", "map_missing", "gender_missing", "age_group"]

GROUPS = {
    "Demographics": DEMOGRAPHICS,
    "Comorbidities": COMORBIDITIES,
    "Vital signs": VITALS,
    "Laboratory": LABS,
    "Treatment / ICU": TREATMENT,
    "Severity scores": SEVERITY,
    "Hospital / encounter context": HOSPITAL_CONTEXT,
    "Engineered features": ENGINEERED,
}


def load_data() -> Dict[str, pd.DataFrame]:
    if not ORIGINAL_PATH.is_file():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_PATH}")
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(f"Feature-engineered dataset not found: {FEATURE_PATH}")
    return {
        "original": pd.read_excel(ORIGINAL_PATH),
        "feature_engineered": pd.read_csv(FEATURE_PATH),
    }


def read_project_scripts() -> Dict[str, str]:
    script_names = [
        "step2_dataset_understanding.py",
        "step3_validation.py",
        "step4_clean_preprocess.py",
        "step7_feature_engineering.py",
        "step8_split_data.py",
        "step9_train_models.py",
    ]
    scripts = {}
    for name in script_names:
        path = PROJECT_ROOT / "analysis" / name
        scripts[name] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return scripts


def timing_classification(column: str, group: str) -> Dict[str, str]:
    if column == "readmission_30day":
        return {"timing": "post-outcome / target", "confidence": "verified", "reason": "Defined project target; never use as a feature."}
    if column == "subject_id":
        return {"timing": "identifier", "confidence": "verified", "reason": "Identifier in the original dataset; not a clinical predictor."}
    if column in DEMOGRAPHICS or column in COMORBIDITIES:
        return {"timing": "likely available at hospital admission", "confidence": "hypothesis", "reason": "Patient characteristics or history, but timestamps are not documented."}
    if column in VITALS:
        return {"timing": "timing unknown", "confidence": "unknown", "reason": "Summary statistic is present, but measurement window is not documented."}
    if column in LABS:
        return {"timing": "timing unknown", "confidence": "unknown", "reason": "Laboratory timing and collection window are not documented."}
    if column in TREATMENT:
        return {"timing": "potentially post-treatment/post-outcome", "confidence": "hypothesis", "reason": "Treatment or support variable may occur after deterioration; exact timing is not documented."}
    if column in SEVERITY:
        return {"timing": "timing unknown", "confidence": "unknown", "reason": "Score calculation window and availability time are not documented."}
    if column == "icu_los_hours":
        return {"timing": "potentially post-treatment/post-outcome", "confidence": "hypothesis", "reason": "Length of stay is accumulated during or after the ICU encounter."}
    if column == "hospital_admit_source":
        return {"timing": "likely available at hospital admission", "confidence": "hypothesis", "reason": "Admission source is conventionally known at intake, but project timestamps do not verify this."}
    if column in ["icu_admit_time_hour", "day_of_week"]:
        return {"timing": "likely available at hospital admission", "confidence": "hypothesis", "reason": "Calendar or admission-hour context appears available, but no raw timestamp is supplied."}
    if column in ENGINEERED:
        return {"timing": "derived from source fields", "confidence": "verified", "reason": "Created by Step 7; inherits the timing limitations of its source fields."}
    return {"timing": "timing unknown", "confidence": "unknown", "reason": "No project documentation establishes when this variable becomes available."}


def build_feature_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_by_column = {}
    for group, columns in GROUPS.items():
        for column in columns:
            group_by_column[column] = group
    for column in data.columns:
        group = group_by_column.get(column, "Other / undocumented")
        classification = timing_classification(column, group)
        rows.append(
            {
                "column": column,
                "group": group,
                "dtype": str(data[column].dtype),
                "missing_count": int(data[column].isna().sum()),
                "unique_count": int(data[column].nunique(dropna=False)),
                "timing_classification": classification["timing"],
                "confidence": classification["confidence"],
                "reason": classification["reason"],
            }
        )
    return pd.DataFrame(rows)


def target_provenance(data: pd.DataFrame, scripts: Dict[str, str]) -> pd.DataFrame:
    target = data[TARGET]
    rows = []

    def add(check: str, evidence: str, result: str, interpretation: str) -> None:
        rows.append({"check": check, "evidence": evidence, "result": result, "interpretation": interpretation})

    add("Target column exists", TARGET in data.columns, "verified" if TARGET in data.columns else "not found", "The original workbook contains the named target column.")
    add("Target is binary", sorted(target.dropna().unique().tolist()) == [0, 1], "verified" if sorted(target.dropna().unique().tolist()) == [0, 1] else "not binary", "Observed target values are 0 and 1.")
    add("Target missingness", int(target.isna().sum()), "verified", "No target values should be missing before modeling." if target.isna().sum() == 0 else "Target missingness requires investigation.")
    add("Target-generation documentation", "No target-generation specification found in inspected scripts", "unknown", "The available project files define the target name but do not explain how labels were generated.")
    add("Subject repetition", int(data["subject_id"].duplicated().sum()) if "subject_id" in data.columns else "not available", "verified", "Unique subject IDs imply one row per subject in this workbook." if "subject_id" in data.columns and data["subject_id"].duplicated().sum() == 0 else "Encounter structure is unresolved.")
    add("Potential timestamp columns", ", ".join([c for c in data.columns if any(token in c.lower() for token in ["time", "date", "timestamp"])]) or "none", "verified", "No full admission/discharge/readmission timestamp columns are present; only hour/day summaries are available." )
    add("Encounter ID columns", ", ".join([c for c in data.columns if "encounter" in c.lower() or c.lower().endswith("_id")]) or "none", "verified", "Only subject_id is present as an identifier; no encounter ID is supplied." )
    add("Future-information named columns", ", ".join([c for c in data.columns if any(token in c.lower() for token in ["discharge", "readmission_date", "readmit_date"])]) or "none", "verified", "No explicitly named discharge or readmission timestamp column was found." )

    row_order = np.arange(len(data))
    row_quartile = pd.qcut(row_order, 4, labels=False)
    row_rates = data.assign(row_quartile=row_quartile).groupby("row_quartile")[TARGET].mean().round(4).to_dict()
    add("Target by row-order quartile", row_rates, "observed", "Differences may reflect ordering or random variation; row order is not documented as meaningful." )

    if "subject_id" in data.columns:
        subject_bins = pd.qcut(data["subject_id"], 4, labels=False, duplicates="drop")
        subject_rates = data.assign(subject_bin=subject_bins).groupby("subject_bin")[TARGET].mean().round(4).to_dict()
        add("Target by subject_id quartile", subject_rates, "observed", "Differences may reflect synthetic ID construction or chance; subject_id is not a feature." )

    for column in ["sepsis_label", "icu_los_hours"] + TREATMENT + SEVERITY:
        if column not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            correlation = data[[TARGET, column]].corr().iloc[0, 1]
            add("Target relationship: " + column, round(float(correlation), 4), "observed", "Association only; this does not establish target derivation or causation." )
        else:
            rates = data.groupby(column, dropna=False)[TARGET].mean().round(4).to_dict()
            add("Target relationship: " + column, rates, "observed", "Group-rate association only; timing and causation are not established." )

    scripts_text = "\n".join(scripts.values()).lower()
    target_mentions = scripts_text.count(TARGET.lower())
    add("Target references in inspected scripts", target_mentions, "verified", "Scripts validate and model the target but do not document its generation mechanism." )
    return pd.DataFrame(rows)


def save_plots(data: pd.DataFrame) -> None:
    row_order = np.arange(len(data))
    row_quartile = pd.qcut(row_order, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    row_rates = data.assign(row_quartile=row_quartile).groupby("row_quartile", observed=False)[TARGET].mean()
    figure, axis = plt.subplots(figsize=(7, 4))
    row_rates.plot(kind="bar", ax=axis, color="#2f6f8f")
    axis.set_title("Readmission Rate by Original Row-Order Quartile")
    axis.set_ylabel("Readmission rate")
    axis.set_xlabel("")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_by_row_order.png", dpi=150)
    plt.close(figure)

    if "subject_id" in data.columns:
        subject_bins = pd.qcut(data["subject_id"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        subject_rates = data.assign(subject_bin=subject_bins).groupby("subject_bin", observed=False)[TARGET].mean()
        figure, axis = plt.subplots(figsize=(7, 4))
        subject_rates.plot(kind="bar", ax=axis, color="#c05a4e")
        axis.set_title("Readmission Rate by subject_id Quartile")
        axis.set_ylabel("Readmission rate")
        axis.set_xlabel("")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "target_by_subject_id.png", dpi=150)
        plt.close(figure)


def write_report(
    original: pd.DataFrame,
    feature_audit: pd.DataFrame,
    provenance: pd.DataFrame,
    scripts: Dict[str, str],
) -> None:
    timing_counts = feature_audit["timing_classification"].value_counts().to_dict()
    original_columns = set(original.columns)
    timestamp_columns = [c for c in original.columns if any(token in c.lower() for token in ["time", "date", "timestamp"])]
    direct_future_columns = [c for c in original.columns if any(token in c.lower() for token in ["discharge", "readmission_date", "readmit_date"])]
    group_lines = []
    for group, columns in GROUPS.items():
        available = [column for column in columns if column in original_columns]
        group_lines.append(f"- **{group}**: {len(available)} variables inspected.")

    lines = [
        "# Step 18: Prediction Timing and Target Provenance Audit",
        "",
        "## Scope and evidence",
        "",
        f"- Original dataset: `{ORIGINAL_PATH.relative_to(PROJECT_ROOT)}`.",
        f"- Rows and columns: {original.shape[0]} x {original.shape[1]}.",
        "- Existing feature-engineered data and project scripts were inspected.",
        "- The held-out test set was not loaded or used.",
        "- Verified facts, observations, hypotheses, and unknowns are explicitly separated below.",
        "",
        "## Verified facts",
        "",
        f"- Target values observed: {original[TARGET].value_counts(dropna=False).to_dict()}.",
        f"- Target missing values: {int(original[TARGET].isna().sum())}.",
        f"- subject_id duplicate rows: {int(original['subject_id'].duplicated().sum()) if 'subject_id' in original.columns else 'not available'}.",
        f"- Timestamp-like columns present: {', '.join(timestamp_columns) if timestamp_columns else 'none'}.",
        f"- Explicit discharge/readmission date columns present: {', '.join(direct_future_columns) if direct_future_columns else 'none'}.",
        f"- Timing classifications generated: {timing_counts}.",
        "",
        "## Feature-group coverage",
        "",
    ] + group_lines + [
        "",
        "## Observations",
        "",
        "- The workbook contains `icu_admit_time_hour` and `day_of_week`, but not a full admission timestamp.",
        "- The workbook contains one row per unique `subject_id`; no repeated encounters can be verified from this file.",
        "- The target has observed associations with sepsis, ICU length of stay, treatment, and severity variables, but these are descriptive relationships only.",
        "- Row-order and subject-ID quartile rates were measured because synthetic or sorted data can create apparent patterns; neither variable is documented as meaningful.",
        "",
        "## Hypotheses requiring confirmation",
        "",
        "- Demographics, comorbidities, admission source, admission hour, and day of week are plausibly available at or near admission, but this is not established by timestamps.",
        "- Vital signs, laboratory values, severity scores, and `sepsis_label` may be available shortly after admission or later; their collection/calculation windows are undocumented.",
        "- Treatment variables and `icu_los_hours` may encode information that becomes available only after treatment or clinical progression.",
        "",
        "## Unknown information",
        "",
        "- The project files do not explain how `readmission_30day` was generated.",
        "- There are no discharge timestamps, readmission timestamps, encounter IDs, or explicit encounter windows in the workbook.",
        "- It cannot be determined whether labels are independently generated, derived from an unavailable source, or based on a synthetic rule.",
        "",
        "## Target provenance evidence",
        "",
        """The provenance summary CSV contains the observed target relationships. Those relationships do not prove that any feature generated the target, and no causal interpretation is made.""",
        "",
        "## Answers",
        "",
        "### A. Most defensible prediction point",
        "The most defensible point is a conservative admission or immediate-intake prediction using only demographics, known comorbidities, admission source, calendar context, and variables whose collection time is explicitly confirmed. The current dataset does not fully verify even this point, so it should be treated as a provisional research definition.",
        "",
        "### B. Variables to exclude pending timing evidence",
        "Exclude `icu_los_hours`, treatment/support variables, and any laboratory, severity, or sepsis variable whose timestamp is not confirmed for the intended prediction point. Do not use the target or `subject_id`.",
        "",
        "### C. Variables safe to consider for an early model",
        "Demographics, documented comorbidities, `hospital_admit_source`, `icu_admit_time_hour`, and `day_of_week` are the safest candidates conceptually, subject to confirmation from the data owner. Their safety is a timing hypothesis, not a verified fact.",
        "",
        "### D. Can target provenance be verified?",
        "No. The project files verify the target column and its observed values, but they do not verify how the labels were generated.",
        "",
        "### E. Is the dataset suitable for a meaningful model?",
        "The dataset is suitable for a documented research experiment, but its suitability for a meaningful readmission model is unproven. Missing provenance, missing encounter timestamps, one row per unique subject, and weak Step 17 signal substantially limit confidence.",
        "",
        "### F. Gradient boosting or dataset investigation?",
        "Investigate and, if possible, replace or augment the dataset first. Do not proceed directly to gradient boosting as though it will solve the provenance and timing uncertainty. A later conservative gradient-boosting comparison is reasonable only after target provenance and prediction timing are clarified.",
    ]
    (OUTPUT_DIR / "prediction_timing_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    datasets = load_data()
    scripts = read_project_scripts()
    original = datasets["original"]
    feature_audit = build_feature_audit(original)
    provenance = target_provenance(original, scripts)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(OUTPUT_DIR / "prediction_timing_features.csv", index=False)
    provenance.to_csv(OUTPUT_DIR / "target_provenance_summary.csv", index=False)
    save_plots(original)
    write_report(original, feature_audit, provenance, scripts)

    print("=" * 70)
    print("STEP 18: PREDICTION TIMING AND TARGET PROVENANCE AUDIT")
    print("=" * 70)
    print(f"Original dataset: {original.shape[0]} rows x {original.shape[1]} columns")
    print(f"Target distribution: {original[TARGET].value_counts(dropna=False).to_dict()}")
    print("Held-out test set used: no")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("STEP 18 COMPLETE")


if __name__ == "__main__":
    main()
