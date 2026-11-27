"""Step 4: clean the dataset without fitting model-dependent transforms."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
TARGET = "readmission_30day"


EXPECTED_GENDER_VALUES = {"F", "M"}
DROP_COLUMNS = {"subject_id", "mechanical_ventilation"}


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    return pd.read_excel(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {TARGET, "gender"}
    missing_required = required_columns.difference(df.columns)
    if missing_required:
        raise ValueError(f"Required columns are missing: {sorted(missing_required)}")

    cleaned = df.copy()
    cleaned["gender"] = cleaned["gender"].where(
        cleaned["gender"].isin(EXPECTED_GENDER_VALUES)
    )

    columns_to_drop = sorted(DROP_COLUMNS.intersection(cleaned.columns))
    cleaned = cleaned.drop(columns=columns_to_drop)

    if not set(cleaned[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1: {TARGET}")

    return cleaned


def main() -> None:
    raw = load_raw_data()
    cleaned = clean_data(raw)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)

    print("=" * 70)
    print("STEP 4: DATA CLEANING AND PREPROCESSING")
    print("=" * 70)
    print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"Cleaned shape: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns")
    print(f"Dropped columns: {sorted(DROP_COLUMNS.intersection(raw.columns))}")
    print(f"Invalid gender values converted to missing: {int(raw['gender'].eq('Mael').sum())}")
    print(f"Remaining missing values: {int(cleaned.isna().sum().sum())}")
    print(f"Saved cleaned dataset: {CLEAN_DATA_PATH}")
    print("\nNote: missing-value imputation and categorical encoding belong in the train-only ML pipeline.")
    print("STEP 4 COMPLETE")


if __name__ == "__main__":
    main()
