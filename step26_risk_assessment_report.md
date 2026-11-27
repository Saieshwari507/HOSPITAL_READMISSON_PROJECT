# Step 26: Final Patient Risk-Assessment Pipeline

## Approved inputs

- `race`
- `gender`
- `age`
- `admission_type_id`
- `admission_source_id`
- `payer_code`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`

## Pipeline steps

1. Validate raw patient inputs and reject forbidden/leakage fields.
2. Apply the Step 22 training-fitted preprocessor.
3. Score with the Step 24 final model.
4. Apply the Step 24 selected threshold.
5. Return estimated probability, risk category, and model-attributed factors.

## Safe example patient

{
  "race": "AfricanAmerican",
  "gender": "Female",
  "age": 75,
  "admission_type_id": 1,
  "admission_source_id": 7,
  "payer_code": "MC",
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2
}

## Example output

{
  "estimated_probability": 0.1819917325377594,
  "selected_threshold": 0.15,
  "crosses_screening_threshold": true,
  "risk_category": "Higher estimated risk",
  "predicted_high_risk": true,
  "model_attributed_factors": {
    "toward_higher_estimated_risk": [
      "number_inpatient",
      "age",
      "number_emergency",
      "admission_source_id_7",
      "admission_type_id_6"
    ],
    "toward_lower_estimated_risk": [
      "number_outpatient",
      "payer_code_Missing",
      "admission_source_id_17",
      "admission_source_id_3",
      "payer_code_OG"
    ]
  },
  "disclaimer": "This tool provides an estimated risk based on patterns in historical data and is intended for academic decision-support demonstration only. It is not a diagnosis or a substitute for professional clinical judgment."
}

## Notes

- Risk categories describe estimated screening risk, not diagnoses.
- The pipeline does not require `patient_nbr` or `encounter_id`.

STEP 26 STATUS: COMPLETE