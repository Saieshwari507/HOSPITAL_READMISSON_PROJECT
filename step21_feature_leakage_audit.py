"""Step 21: select prediction-time features without training a model."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "uci_diabetes_130_us_hospitals" / "diabetic_data.csv"
DECISIONS_PATH = PROJECT_ROOT / "analysis" / "step21_feature_decisions.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "analysis" / "step21_modeling_feature_list.csv"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step21_feature_leakage_report.md"
TARGET = "readmitted"
PREDICTION_POINT = "Admission / early hospitalization prediction using information available at or near the beginning of the encounter."

SAFE_FEATURES = {
    "race": ("Documented demographic characteristic; reasonably available at intake.", "No documented timing concern for this field."),
    "gender": ("Documented demographic characteristic; reasonably available at intake.", "No documented timing concern for this field."),
    "age": ("Demographic age band available at intake.", "Age is grouped rather than exact; source timing is not timestamped."),
    "admission_type_id": ("Admission classification is recorded at admission.", "Exact local coding definitions require the UCI mapping/documentation."),
    "admission_source_id": ("Admission source is an administrative intake attribute.", "Exact local coding definitions require the UCI mapping/documentation."),
    "payer_code": ("Administrative payer information is generally available during intake.", "Missingness is substantial and availability timing is not explicitly timestamped."),
    "number_outpatient": ("Prior outpatient utilization count is explicitly described as prior utilization.", "Lookback window and exact extraction timing should be documented."),
    "number_emergency": ("Prior emergency utilization count is explicitly described as prior utilization.", "Lookback window and exact extraction timing should be documented."),
    "number_inpatient": ("Prior inpatient utilization count is explicitly described as prior utilization.", "Lookback window and exact extraction timing should be documented."),
}

IDENTIFIERS = {
    "encounter_id": "Encounter identifier; required for linkage but not a clinical predictor.",
    "patient_nbr": "Patient identifier; required for grouped splitting but not a clinical predictor.",
}

TARGET_COLUMNS = {"readmitted": "Original three-level readmission target; excluded from features."}

POTENTIAL_LEAKAGE = {
    "discharge_disposition_id": ("Discharge disposition is known at discharge and can encode the discharge course.", "Unavailable at admission/early prediction."),
    "time_in_hospital": ("Length of stay is accumulated during the encounter and is only known after discharge.", "Direct post-admission information."),
    "num_lab_procedures": ("Count can accumulate during the hospitalization.", "Collection timing is not documented."),
    "num_procedures": ("Procedure count can accumulate during the hospitalization.", "May include procedures after the prediction point."),
    "num_medications": ("Medication count can accumulate during the hospitalization.", "May include treatment after the prediction point."),
    "max_glu_serum": ("Maximum glucose result may summarize tests collected during the encounter.", "Collection window is not documented."),
    "A1Cresult": ("HbA1c result availability and collection timing are not established.", "May be ordered or resulted after admission."),
    "change": ("Medication-change status describes treatment changes during the encounter.", "Potentially post-treatment."),
    "diabetesMed": ("Diabetes medication status may reflect inpatient treatment decisions.", "Potentially post-treatment."),
}

UNCERTAIN_TIMING = {
    "weight": ("Weight is described, but collection timing is not documented.", "May be available at admission, but cannot be verified."),
    "medical_specialty": ("Specialty field timing and whether it reflects final service assignment are not documented.", "May change or be finalized after admission."),
    "diag_1": ("Diagnosis code may be present at admission but coding timing is not documented.", "Could be provisional, coded later, or finalized during the stay."),
    "diag_2": ("Diagnosis code timing is not documented.", "May be secondary or finalized after admission."),
    "diag_3": ("Diagnosis code timing is not documented.", "May be secondary or finalized after admission."),
    "number_diagnoses": ("The count may reflect coding completed during the encounter.", "Availability at the prediction point is not documented."),
}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Selected dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)


def classify_columns(data: pd.DataFrame) -> pd.DataFrame:
    decisions = []
    for column in data.columns:
        if column in IDENTIFIERS:
            category = "EXCLUDE_IDENTIFIER"
            decision, concern = IDENTIFIERS[column], "Identifier is not a prediction-time clinical feature."
        elif column in TARGET_COLUMNS:
            category = "EXCLUDE_TARGET"
            decision, concern = TARGET_COLUMNS[column], "Target must not enter the feature matrix."
        elif column in SAFE_FEATURES:
            category = "SAFE_FOR_MODELING"
            decision, concern = SAFE_FEATURES[column]
        elif column in POTENTIAL_LEAKAGE or column in MEDICATION_COLUMNS:
            category = "POTENTIAL_LEAKAGE"
            if column in POTENTIAL_LEAKAGE:
                decision, concern = POTENTIAL_LEAKAGE[column]
            else:
                decision = "Medication field may reflect treatment during the encounter."
                concern = "Availability and timing relative to admission are not documented."
        elif column in UNCERTAIN_TIMING:
            category = "UNCERTAIN_TIMING"
            decision, concern = UNCERTAIN_TIMING[column]
        else:
            category = "UNCERTAIN_TIMING"
            decision = "No documented evidence establishes availability at the selected prediction point."
            concern = "Conservatively excluded until timing is documented."
        decisions.append(
            {
                "column": column,
                "category": category,
                "decision": decision,
                "reason": decision,
                "timing_concern": concern,
                "dtype": str(data[column].dtype),
                "missing_count": int(data[column].isna().sum()),
                "missing_percentage": round(data[column].isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(decisions)


def write_report(data: pd.DataFrame, decisions: pd.DataFrame, modeling_features: list) -> None:
    grouped = decisions.groupby("category")["column"].apply(list).to_dict()
    lines = [
        "# Step 21: Prediction-Time Feature Selection and Leakage Control",
        "",
        "## 1. Prediction point",
        "",
        PREDICTION_POINT,
        "",
        "Because exact feature timestamps are not documented, this audit uses conservative exclusions.",
        "",
        "## 2. Target definition",
        "",
        "The original `readmitted` target is retained only as the label source. The final target transformation is `readmitted == '<30'`. It is excluded from the feature list.",
        "",
        "## 3. Safe features",
        "",
    ]
    lines.extend(f"- `{column}`" for column in grouped.get("SAFE_FOR_MODELING", []))
    lines.extend(["", "## 4. Excluded identifiers", ""])
    lines.extend(f"- `{column}`" for column in grouped.get("EXCLUDE_IDENTIFIER", []))
    lines.extend(["", "## 5. Excluded target", ""])
    lines.extend(f"- `{column}`" for column in grouped.get("EXCLUDE_TARGET", []))
    lines.extend(["", "## 6. Potential leakage", ""])
    lines.extend(f"- `{column}`: excluded because its timing may be during or after the encounter." for column in grouped.get("POTENTIAL_LEAKAGE", []))
    lines.extend(["", "## 7. Uncertain-timing features", ""])
    lines.extend(f"- `{column}`: excluded because availability at the prediction point is not documented." for column in grouped.get("UNCERTAIN_TIMING", []))
    lines.extend([
        "",
        "## 8. Final modeling feature set",
        "",
    ])
    lines.extend(f"- `{column}`" for column in modeling_features)
    lines.extend([
        "",
        "## 9. Remaining limitations",
        "",
        "- The selected prediction point is an academic design assumption, not a timestamp proven by the flat file.",
        "- Prior utilization counts are treated as safe because UCI describes them as prior-year counts; the exact extraction process should still be recorded.",
        "- Admission/early availability for payer and admission coding is plausible but not timestamp-verified.",
        "- Diagnosis codes are excluded because coding timing is not established.",
        "- No model was trained or evaluated in Step 21.",
        "",
        "## Decision",
        "",
        "FINAL MODELING FEATURE SET APPROVED: YES",
        "",
        "The feature set is approved for the next modeling stage as a conservative research schema. Approval means the columns have passed this timing/leakage screen; it does not establish clinical validity or model performance.",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_data()
    decisions = classify_columns(data)
    modeling_features = decisions.loc[
        decisions["category"] == "SAFE_FOR_MODELING", "column"
    ].tolist()
    decisions.to_csv(DECISIONS_PATH, index=False)
    pd.DataFrame({"feature": modeling_features}).to_csv(FEATURE_LIST_PATH, index=False)
    write_report(data, decisions, modeling_features)

    print("=" * 70)
    print("STEP 21: PREDICTION-TIME FEATURE SELECTION")
    print("=" * 70)
    print(f"Prediction point: {PREDICTION_POINT}")
    print(f"Final modeling features ({len(modeling_features)}): {modeling_features}")
    print("Excluded features by category:")
    for category, group in decisions.groupby("category"):
        if category != "SAFE_FOR_MODELING":
            print(f"  {category}: {len(group)}")
    print("Models trained: no")
    print("FINAL MODELING FEATURE SET APPROVED: YES")
    print(f"Saved decisions: {DECISIONS_PATH}")
    print(f"Saved feature list: {FEATURE_LIST_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print("STEP 21 COMPLETE")


if __name__ == "__main__":
    main()
