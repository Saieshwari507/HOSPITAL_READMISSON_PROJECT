"""Step 14: define a validated new-patient risk assessment interface."""

from pathlib import Path

import joblib
import pandas as pd

try:
    from .step7_feature_engineering import add_features
except ImportError:
    from step7_feature_engineering import add_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "assessment_outputs"
TARGET = "readmission_30day"
TOP_FACTORS = 5

ENGINEERED_COLUMNS = {
    "comorbidity_count",
    "treatment_flag_count",
    "lab_missing_count",
    "bmi_missing",
    "map_missing",
    "gender_missing",
    "age_group",
}


def load_model_and_schema():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run Step 9 first.")
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}. Run Step 8 first.")

    model = joblib.load(MODEL_PATH)
    train_data = pd.read_csv(TRAIN_PATH)
    expected_columns = list(train_data.drop(columns=[TARGET]).columns)
    return model, expected_columns


def assess_patient(patient_data: dict) -> dict:
    """Return an estimated readmission probability and model-derived factors."""
    model, expected_columns = load_model_and_schema()
    patient_frame = pd.DataFrame([patient_data])
    patient_frame = add_features(patient_frame)

    missing_columns = [
        column for column in expected_columns if column not in patient_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Patient input is missing required fields: " + ", ".join(missing_columns)
        )

    patient_frame = patient_frame[expected_columns]
    probability = float(model.predict_proba(patient_frame)[0, 1])
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    transformed = preprocessor.transform(patient_frame)
    feature_names = list(preprocessor.get_feature_names_out())
    contributions = transformed[0] * classifier.coef_[0]
    factor_data = pd.DataFrame(
        {"feature": feature_names, "contribution": contributions}
    ).sort_values("contribution", ascending=False)

    positive_factors = factor_data.head(TOP_FACTORS)["feature"].tolist()
    negative_factors = factor_data.tail(TOP_FACTORS)["feature"].tolist()
    baseline = 0.0538
    if probability < baseline:
        relative_band = "below historical baseline"
    elif probability < baseline * 2:
        relative_band = "near historical baseline"
    else:
        relative_band = "above historical baseline"

    return {
        "estimated_probability": probability,
        "relative_band": relative_band,
        "positive_model_factors": positive_factors,
        "negative_model_factors": negative_factors,
        "disclaimer": (
            "Experimental model output for research only; not a diagnosis or "
            "clinical recommendation."
        ),
    }


def main() -> None:
    model, expected_columns = load_model_and_schema()
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}. Run Step 8 first.")

    test_data = pd.read_csv(TEST_PATH)
    raw_columns = [
        column
        for column in expected_columns
        if column not in ENGINEERED_COLUMNS
    ]
    example_patient = test_data.iloc[0][raw_columns].to_dict()
    result = assess_patient(example_patient)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"field": raw_columns}).to_json(
        OUTPUT_DIR / "new_patient_schema.json", orient="records", indent=2
    )

    print("=" * 70)
    print("STEP 14: NEW-PATIENT RISK ASSESSMENT")
    print("=" * 70)
    print(f"Required input fields: {len(raw_columns)}")
    print(f"Example estimated probability: {result['estimated_probability']:.4f}")
    print(f"Relative band: {result['relative_band']}")
    print(f"Schema saved to: {OUTPUT_DIR / 'new_patient_schema.json'}")
    print("\nThe assessment returns model signals, not clinical advice.")
    print("STEP 14 COMPLETE")


if __name__ == "__main__":
    main()
