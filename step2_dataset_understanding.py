"""Step 2: inspect the hospital readmission dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"
TARGET = "readmission_30day"
IDENTIFIER = "subject_id"


def load_dataset() -> pd.DataFrame:
    """Load the project dataset from its fixed project-relative location."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_excel(DATA_PATH)


def main() -> None:
    df = load_dataset()

    print("=" * 70)
    print("STEP 2: DATASET UNDERSTANDING")
    print("=" * 70)
    print(f"Dataset path: {DATA_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nColumn inventory:")
    for number, column in enumerate(df.columns, start=1):
        print(
            f"{number:2}. {column:<35} "
            f"dtype={str(df[column].dtype):<8} "
            f"missing={df[column].isna().sum():4} "
            f"unique={df[column].nunique(dropna=False):4}"
        )

    print("\nCategorical values:")
    for column in df.select_dtypes(include=["object", "category", "string"]):
        values = sorted(df[column].dropna().astype(str).unique())
        print(f"{column}: {values}")

    print("\nMissing values:")
    missing = df.isna().sum().loc[lambda values: values.gt(0)]
    if missing.empty:
        print("No missing values found.")
    else:
        for column, count in missing.sort_values(ascending=False).items():
            percentage = count / len(df) * 100
            print(f"{column}: {count} ({percentage:.2f}%)")

    print("\nTarget and identifier checks:")
    if TARGET not in df.columns:
        raise ValueError(f"Required target column not found: {TARGET}")
    print(f"{TARGET} counts: {df[TARGET].value_counts(dropna=False).to_dict()}")

    if IDENTIFIER in df.columns:
        print(f"{IDENTIFIER} unique values: {df[IDENTIFIER].nunique()}")
        print(f"Duplicate {IDENTIFIER} rows: {df[IDENTIFIER].duplicated().sum()}")

    print("\nSTEP 2 COMPLETE")


if __name__ == "__main__":
    main()
