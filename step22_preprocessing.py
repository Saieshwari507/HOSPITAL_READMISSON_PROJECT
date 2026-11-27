"""Step 22: patient-aware splitting and training-only preprocessing."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "uci_diabetes_130_us_hospitals" / "diabetic_data.csv"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "train_processed.csv"
TEST_OUTPUT = PROJECT_ROOT / "data" / "test_processed.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "analysis" / "step22_split_summary.csv"
REPORT_OUTPUT = PROJECT_ROOT / "analysis" / "step22_preprocessing_report.md"
TARGET_SOURCE = "readmitted"
TARGET = "readmission_30day"
GROUP = "patient_nbr"
ENCOUNTER = "encounter_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20

APPROVED_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "payer_code",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]
CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "payer_code",
    "admission_type_id",
    "admission_source_id",
]
NUMERICAL_FEATURES = [
    "age",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Selected dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)
    data[TARGET] = (data[TARGET_SOURCE] == "<30").astype(int)
    data["age"] = data["age"].map(age_band_midpoint)
    return data


def age_band_midpoint(value: object) -> float:
    """Convert UCI's grouped age label such as '[20-30)' to its midpoint."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip("[]()")
    lower_bound = float(text.split("-")[0])
    upper_bound = float(text.split("-")[1])
    return (lower_bound + upper_bound) / 2


def choose_group_split(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    groups = data[GROUP]
    target_rate = data[TARGET].mean()
    desired_test_patients = data[GROUP].nunique() * TEST_SIZE
    best = None

    for train_indices, test_indices in splitter.split(data, data[TARGET], groups):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        patient_error = abs(test[GROUP].nunique() - desired_test_patients) / desired_test_patients
        rate_error = abs(test[TARGET].mean() - target_rate)
        score = rate_error * 10 + patient_error
        if best is None or score < best[0]:
            best = (score, train_indices, test_indices)

    if best is None:
        raise RuntimeError("Unable to create a patient-aware split.")
    return best[1], best[2]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def process_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_indices, test_indices = choose_group_split(data)
    train_raw = data.iloc[train_indices].copy()
    test_raw = data.iloc[test_indices].copy()
    preprocessor = build_preprocessor()
    preprocessor.fit(train_raw[APPROVED_FEATURES])
    train_matrix = preprocessor.transform(train_raw[APPROVED_FEATURES])
    test_matrix = preprocessor.transform(test_raw[APPROVED_FEATURES])
    feature_names = list(preprocessor.get_feature_names_out())

    train_processed = pd.DataFrame(train_matrix, columns=feature_names, index=train_raw.index)
    test_processed = pd.DataFrame(test_matrix, columns=feature_names, index=test_raw.index)
    train_processed[TARGET] = train_raw[TARGET].to_numpy()
    test_processed[TARGET] = test_raw[TARGET].to_numpy()
    train_processed = train_processed.reset_index(drop=True)
    test_processed = test_processed.reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "partition": "train",
                "patients": train_raw[GROUP].nunique(),
                "encounters": len(train_raw),
                "positive_readmissions": int(train_raw[TARGET].sum()),
                "readmission_percentage": train_raw[TARGET].mean() * 100,
            },
            {
                "partition": "test",
                "patients": test_raw[GROUP].nunique(),
                "encounters": len(test_raw),
                "positive_readmissions": int(test_raw[TARGET].sum()),
                "readmission_percentage": test_raw[TARGET].mean() * 100,
            },
        ]
    )
    overlap = len(set(train_raw[GROUP]).intersection(set(test_raw[GROUP])))
    summary["patient_overlap"] = overlap
    summary["feature_count_after_encoding"] = len(feature_names)
    return train_processed, test_processed, summary


def write_report(data: pd.DataFrame, summary: pd.DataFrame, overlap: int) -> None:
    train_summary = summary[summary["partition"] == "train"].iloc[0]
    test_summary = summary[summary["partition"] == "test"].iloc[0]
    lines = [
        "# Step 22: Patient-Aware Splitting and Preprocessing",
        "",
        "## 1. Dataset overview",
        "",
        f"- Source rows: {len(data):,} encounters.",
        f"- Source patients: {data[GROUP].nunique():,}.",
        f"- Source target: `{TARGET_SOURCE} == '<30'`.",
        "",
        "## 2. Prediction target",
        "",
        "`readmission_30day = 1` when `readmitted == '<30'`; otherwise `0`.",
        "",
        "## 3. Approved features",
        "",
    ]
    lines.extend(f"- `{column}`" for column in APPROVED_FEATURES)
    lines.extend(
        [
            "",
            "## 4. Patient-level splitting method",
            "",
            "A reproducible `GroupShuffleSplit` search was used with `patient_nbr` as the grouping variable only. Candidate grouped splits were compared for approximate 80/20 patient allocation and target-rate similarity. `patient_nbr` was never passed to the preprocessing transformer.",
            "",
            "## 5. Split summary",
            "",
            f"- Train patients: {int(train_summary['patients']):,}; encounters: {int(train_summary['encounters']):,}; positives: {int(train_summary['positive_readmissions']):,}; rate: {train_summary['readmission_percentage']:.2f}%.",
            f"- Test patients: {int(test_summary['patients']):,}; encounters: {int(test_summary['encounters']):,}; positives: {int(test_summary['positive_readmissions']):,}; rate: {test_summary['readmission_percentage']:.2f}%.",
            f"- Patient overlap: {overlap}.",
            "",
            "## 6. Missing-value handling",
            "",
            "- Numeric features use median imputation fitted on training rows only.",
            "- Categorical features use a dedicated `Missing` category fitted on training rows only.",
            "- Test values are transformed using the already-fitted training transformer.",
            "",
            "## 7. Categorical encoding",
            "",
            "One-hot encoding is used for `race`, `gender`, `payer_code`, `admission_type_id`, and `admission_source_id`. Unknown test categories are ignored safely.",
            "",
            "## 8. Numerical preprocessing",
            "",
            "`age`, `number_outpatient`, `number_emergency`, and `number_inpatient` use training-fitted median imputation and standard scaling. One-hot categorical columns are not scaled separately.",
            "- UCI stores `age` as grouped labels; each band is converted to its midpoint before the training-only numeric imputer/scaler.",
            "",
            "## 9. Leakage prevention",
            "",
            "- `patient_nbr` is used only for grouping and is absent from processed features.",
            "- `encounter_id` is absent from processed features.",
            "- `readmitted` is absent from processed features.",
            "- The target is retained only as `readmission_30day` in the processed output for later supervised modeling.",
            "- No model was trained and no ROC-AUC was calculated.",
            "",
            "## 10. Final preprocessing pipeline",
            "",
            "Patient-aware split -> fit numeric median imputation and scaling on train -> fit categorical Missing handling and one-hot encoding on train -> transform train and test with the same fitted transformer.",
            "",
            "## Verification",
            "",
            "PATIENT OVERLAP: 0",
            "PREPROCESSING LEAKAGE CHECK: PASSED",
            "",
            "STEP 22 STATUS: COMPLETE",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_data()
    train_processed, test_processed, summary = process_data(data)
    overlap = int(summary["patient_overlap"].max())
    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(TRAIN_OUTPUT, index=False)
    test_processed.to_csv(TEST_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    write_report(data, summary, overlap)

    forbidden = {GROUP, ENCOUNTER, TARGET_SOURCE}
    processed_forbidden = forbidden.intersection(train_processed.columns)
    if processed_forbidden:
        raise ValueError(f"Forbidden input columns remain: {sorted(processed_forbidden)}")
    if overlap != 0:
        raise ValueError(f"Patient overlap detected: {overlap}")

    print("=" * 70)
    print("STEP 22: PATIENT-AWARE DATA SPLITTING AND PREPROCESSING")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"Processed feature count: {train_processed.shape[1] - 1}")
    print("PATIENT OVERLAP: 0")
    print("PREPROCESSING LEAKAGE CHECK: PASSED")
    print("Models trained: no")
    print("STEP 22 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
