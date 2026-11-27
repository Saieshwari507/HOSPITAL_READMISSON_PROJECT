"""Step 26: reusable patient risk-assessment pipeline for approved Step 21 features."""

from pathlib import Path
import json
import re
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step22_preprocessing import APPROVED_FEATURES, age_band_midpoint  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "final_model_config.joblib"
SCHEMA_PATH = PROJECT_ROOT / "analysis" / "step26_prediction_schema.json"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step26_risk_assessment_report.md"

DISCLAIMER = (
    "This tool provides an estimated risk based on patterns in historical data and is "
    "intended for academic decision-support demonstration only. It is not a diagnosis or "
    "a substitute for professional clinical judgment."
)

FORBIDDEN_FIELDS = {
    "patient_nbr",
    "encounter_id",
    "readmitted",
    "readmission_30day",
}

ALLOWED_RACE = {
    "Caucasian",
    "AfricanAmerican",
    "Asian",
    "Hispanic",
    "Other",
    "Missing",
}
ALLOWED_GENDER = {"Female", "Male", "Unknown/Invalid"}
AGE_BAND_PATTERN = re.compile(r"^\[\d+-\d+\)$")


def load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Final model not found. Run Step 24 first.")
    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError("Preprocessor not found. Run Step 24 first.")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("Final model config not found. Run Step 24 first.")
    return (
        joblib.load(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(CONFIG_PATH),
    )


def normalize_age(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("age is required.")
    if isinstance(value, (int, float)):
        age = float(value)
        if age < 0 or age > 120:
            raise ValueError("age must be between 0 and 120.")
        return age
    text = str(value).strip()
    if AGE_BAND_PATTERN.match(text):
        return age_band_midpoint(text)
    raise ValueError("age must be a numeric year value or a UCI-style band such as '[70-80)'.")


def validate_patient_input(patient: Dict[str, Any]) -> Dict[str, Any]:
    unknown_forbidden = FORBIDDEN_FIELDS.intersection(patient.keys())
    if unknown_forbidden:
        raise ValueError(
            "Forbidden fields are not accepted: " + ", ".join(sorted(unknown_forbidden))
        )

    missing = [field for field in APPROVED_FEATURES if field not in patient or patient[field] in (None, "")]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    cleaned = {}
    race = str(patient["race"]).strip()
    if race not in ALLOWED_RACE:
        raise ValueError(f"Invalid race value: {race}")
    cleaned["race"] = race

    gender = str(patient["gender"]).strip()
    if gender not in ALLOWED_GENDER:
        raise ValueError(f"Invalid gender value: {gender}")
    cleaned["gender"] = gender

    cleaned["age"] = normalize_age(patient["age"])

    for field in ("admission_type_id", "admission_source_id"):
        try:
            value = int(patient[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be an integer.") from error
        if value < 0:
            raise ValueError(f"{field} must be non-negative.")
        cleaned[field] = value

    cleaned["payer_code"] = str(patient["payer_code"]).strip()
    if not cleaned["payer_code"]:
        raise ValueError("payer_code is required.")

    for field in ("number_outpatient", "number_emergency", "number_inpatient"):
        try:
            value = int(patient[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be a non-negative integer.") from error
        if value < 0:
            raise ValueError(f"{field} must be non-negative.")
        cleaned[field] = value

    return cleaned


def model_attributed_factors(model, feature_names, processed_row: pd.Series) -> Dict[str, List[str]]:
    if hasattr(model, "coef_"):
        contributions = processed_row.to_numpy() * model.coef_[0]
    else:
        centered = processed_row.to_numpy() - processed_row.mean()
        contributions = model.feature_importances_ * centered

    frame = pd.DataFrame({"feature": feature_names, "contribution": contributions})
    frame["label"] = frame["feature"].str.replace("numeric__", "", regex=False).str.replace(
        "categorical__", "", regex=False
    )
    higher = frame.sort_values("contribution", ascending=False).head(5)["label"].tolist()
    lower = frame.sort_values("contribution", ascending=True).head(5)["label"].tolist()
    return {
        "toward_higher_estimated_risk": higher,
        "toward_lower_estimated_risk": lower,
    }


def assess_patient(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return estimated probability, screening decision, and model-attributed factors."""
    model, preprocessor, config = load_artifacts()
    patient = validate_patient_input(patient_data)
    threshold = float(config["threshold"])
    feature_names = list(config["feature_names"])

    frame = pd.DataFrame([patient])[APPROVED_FEATURES]
    matrix = preprocessor.transform(frame)
    processed_frame = pd.DataFrame(matrix, columns=feature_names)
    processed = processed_frame.iloc[0]
    probability = float(model.predict_proba(processed_frame)[0, 1])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Model returned an invalid probability.")

    crosses_threshold = probability >= threshold
    if crosses_threshold:
        risk_category = "Higher estimated risk"
        predicted_high_risk = True
    else:
        risk_category = "Lower estimated risk"
        predicted_high_risk = False

    factors = model_attributed_factors(model, feature_names, processed)
    return {
        "estimated_probability": probability,
        "selected_threshold": threshold,
        "crosses_screening_threshold": crosses_threshold,
        "risk_category": risk_category,
        "predicted_high_risk": predicted_high_risk,
        "model_attributed_factors": factors,
        "disclaimer": DISCLAIMER,
    }


def example_patient() -> Dict[str, Any]:
    return {
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": 75,
        "admission_type_id": 1,
        "admission_source_id": 7,
        "payer_code": "MC",
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
    }


def write_schema() -> None:
    schema = {
        "required_fields": APPROVED_FEATURES,
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "field_types": {
            "race": "categorical string",
            "gender": "categorical string",
            "age": "numeric years or UCI age band",
            "admission_type_id": "integer",
            "admission_source_id": "integer",
            "payer_code": "string",
            "number_outpatient": "non-negative integer",
            "number_emergency": "non-negative integer",
            "number_inpatient": "non-negative integer",
        },
        "output_fields": [
            "estimated_probability",
            "selected_threshold",
            "crosses_screening_threshold",
            "risk_category",
            "predicted_high_risk",
            "model_attributed_factors",
            "disclaimer",
        ],
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_report(example_result: Dict[str, Any]) -> None:
    lines = [
        "# Step 26: Final Patient Risk-Assessment Pipeline",
        "",
        "## Approved inputs",
        "",
    ]
    lines.extend(f"- `{field}`" for field in APPROVED_FEATURES)
    lines.extend(
        [
            "",
            "## Pipeline steps",
            "",
            "1. Validate raw patient inputs and reject forbidden/leakage fields.",
            "2. Apply the Step 22 training-fitted preprocessor.",
            "3. Score with the Step 24 final model.",
            "4. Apply the Step 24 selected threshold.",
            "5. Return estimated probability, risk category, and model-attributed factors.",
            "",
            "## Safe example patient",
            "",
            json.dumps(example_patient(), indent=2),
            "",
            "## Example output",
            "",
            json.dumps(example_result, indent=2),
            "",
            "## Notes",
            "",
            "- Risk categories describe estimated screening risk, not diagnoses.",
            "- The pipeline does not require `patient_nbr` or `encounter_id`.",
            "",
            "STEP 26 STATUS: COMPLETE",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_schema()
    result = assess_patient(example_patient())
    write_report(result)
    print("=" * 70)
    print("STEP 26: FINAL PATIENT RISK-ASSESSMENT PIPELINE")
    print("=" * 70)
    print(f"Example probability: {result['estimated_probability']:.4f}")
    print(f"Risk category: {result['risk_category']}")
    print("STEP 26 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
