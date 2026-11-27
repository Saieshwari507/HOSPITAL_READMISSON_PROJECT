"""Step 7: create deterministic features for later model training."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "feature_engineered_dataset.csv"
TARGET = "readmission_30day"

COMORBIDITY_COLUMNS = [
    "diabetes",
    "hypertension",
    "chf",
    "copd",
    "chronic_kidney_disease",
    "liver_disease",
    "immunosuppression",
    "cad",
    "atrial_fibrillation",
    "cancer_active",
]
TREATMENT_FLAG_COLUMNS = [
    "vasopressors_flag",
    "mechanical_ventilation",
    "antibiotics_24h",
    "insulin_infusion_flag",
]
LAB_COLUMNS = [
    "wbc",
    "lactate_mmol",
    "creatinine",
    "platelet_count",
    "bilirubin_total",
    "glucose",
    "ph_arterial",
    "pao2_fio2_ratio",
    "inr",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "hematocrit",
    "hemoglobin",
]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}. Run Step 4 first."
        )
    return pd.read_csv(INPUT_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    available_comorbidities = [
        column for column in COMORBIDITY_COLUMNS if column in engineered.columns
    ]
    available_treatments = [
        column for column in TREATMENT_FLAG_COLUMNS if column in engineered.columns
    ]
    available_labs = [column for column in LAB_COLUMNS if column in engineered.columns]

    if available_comorbidities:
        engineered["comorbidity_count"] = engineered[available_comorbidities].sum(axis=1)
    if available_treatments:
        engineered["treatment_flag_count"] = engineered[available_treatments].sum(axis=1)
    if available_labs:
        engineered["lab_missing_count"] = engineered[available_labs].isna().sum(axis=1)

    if "bmi" in engineered.columns:
        engineered["bmi_missing"] = engineered["bmi"].isna().astype(int)
    if "map_mean" in engineered.columns:
        engineered["map_missing"] = engineered["map_mean"].isna().astype(int)
    if "gender" in engineered.columns:
        engineered["gender_missing"] = engineered["gender"].isna().astype(int)
    if "age" in engineered.columns:
        engineered["age_group"] = pd.cut(
            pd.to_numeric(engineered["age"], errors="coerce"),
            bins=[0, 40, 55, 65, 75, 120],
            labels=["18-40", "41-55", "56-65", "66-75", "76+"],
            right=True,
        )

    return engineered


def main() -> None:
    source = load_data()
    engineered = add_features(source)
    engineered.to_csv(OUTPUT_PATH, index=False)

    added_columns = [column for column in engineered.columns if column not in source.columns]
    print("=" * 70)
    print("STEP 7: FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Input shape: {source.shape[0]} rows x {source.shape[1]} columns")
    print(f"Output shape: {engineered.shape[0]} rows x {engineered.shape[1]} columns")
    print(f"Added features: {added_columns}")
    print(f"Target preserved: {TARGET in engineered.columns}")
    print(f"Saved feature-engineered dataset: {OUTPUT_PATH}")
    print("\nNote: imputation, encoding, and scaling remain for the train-only pipeline.")
    print("STEP 7 COMPLETE")


if __name__ == "__main__":
    main()
