"""Step 8: create a stratified, leakage-safe train/test split."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_engineered_dataset.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
TARGET = "readmission_30day"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_validate_data() -> Tuple[pd.DataFrame, pd.Series]:
    if not FEATURE_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {FEATURE_DATA_PATH}."
        )
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}.")

    features = pd.read_csv(FEATURE_DATA_PATH)
    raw = pd.read_excel(RAW_DATA_PATH, usecols=["subject_id"])

    if len(features) != len(raw):
        raise ValueError("Feature and subject datasets have different row counts.")
    if TARGET not in features.columns:
        raise ValueError(f"Target column not found: {TARGET}")
    if raw["subject_id"].duplicated().any():
        raise ValueError("Repeated subject IDs require a grouped split.")

    return features, raw["subject_id"]


def main() -> None:
    data, subject_ids = load_and_validate_data()
    row_indices = pd.Series(range(len(data)), name="row_index")

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    train_indices = train_indices.sort_values().to_numpy()
    test_indices = test_indices.sort_values().to_numpy()

    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    train_subjects = set(subject_ids.iloc[train_indices])
    test_subjects = set(subject_ids.iloc[test_indices])

    if train_subjects.intersection(test_subjects):
        raise ValueError("Subject leakage detected between train and test sets.")
    if set(train_indices).intersection(test_indices):
        raise ValueError("Row overlap detected between train and test sets.")

    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print("=" * 70)
    print("STEP 8: LEAKAGE-SAFE TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Random state: {RANDOM_STATE}")
    print(f"Train shape: {train_data.shape[0]} rows x {train_data.shape[1]} columns")
    print(f"Test shape: {test_data.shape[0]} rows x {test_data.shape[1]} columns")
    print(f"Train readmission rate: {train_data[TARGET].mean() * 100:.2f}%")
    print(f"Test readmission rate: {test_data[TARGET].mean() * 100:.2f}%")
    print(f"Train/test subject overlap: {len(train_subjects.intersection(test_subjects))}")
    print(f"Saved train data: {TRAIN_PATH}")
    print(f"Saved test data: {TEST_PATH}")
    print("\nImputation, encoding, and scaling must be fitted on train data only.")
    print("STEP 8 COMPLETE")


if __name__ == "__main__":
    main()
