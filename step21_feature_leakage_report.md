# Step 21: Prediction-Time Feature Selection and Leakage Control

## 1. Prediction point

Admission / early hospitalization prediction using information available at or near the beginning of the encounter.

Because exact feature timestamps are not documented, this audit uses conservative exclusions.

## 2. Target definition

The original `readmitted` target is retained only as the label source. The final target transformation is `readmitted == '<30'`. It is excluded from the feature list.

## 3. Safe features

- `race`
- `gender`
- `age`
- `admission_type_id`
- `admission_source_id`
- `payer_code`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`

## 4. Excluded identifiers

- `encounter_id`
- `patient_nbr`

## 5. Excluded target

- `readmitted`

## 6. Potential leakage

- `discharge_disposition_id`: excluded because its timing may be during or after the encounter.
- `time_in_hospital`: excluded because its timing may be during or after the encounter.
- `num_lab_procedures`: excluded because its timing may be during or after the encounter.
- `num_procedures`: excluded because its timing may be during or after the encounter.
- `num_medications`: excluded because its timing may be during or after the encounter.
- `max_glu_serum`: excluded because its timing may be during or after the encounter.
- `A1Cresult`: excluded because its timing may be during or after the encounter.
- `metformin`: excluded because its timing may be during or after the encounter.
- `repaglinide`: excluded because its timing may be during or after the encounter.
- `nateglinide`: excluded because its timing may be during or after the encounter.
- `chlorpropamide`: excluded because its timing may be during or after the encounter.
- `glimepiride`: excluded because its timing may be during or after the encounter.
- `acetohexamide`: excluded because its timing may be during or after the encounter.
- `glipizide`: excluded because its timing may be during or after the encounter.
- `glyburide`: excluded because its timing may be during or after the encounter.
- `tolbutamide`: excluded because its timing may be during or after the encounter.
- `pioglitazone`: excluded because its timing may be during or after the encounter.
- `rosiglitazone`: excluded because its timing may be during or after the encounter.
- `acarbose`: excluded because its timing may be during or after the encounter.
- `miglitol`: excluded because its timing may be during or after the encounter.
- `troglitazone`: excluded because its timing may be during or after the encounter.
- `tolazamide`: excluded because its timing may be during or after the encounter.
- `examide`: excluded because its timing may be during or after the encounter.
- `citoglipton`: excluded because its timing may be during or after the encounter.
- `insulin`: excluded because its timing may be during or after the encounter.
- `glyburide-metformin`: excluded because its timing may be during or after the encounter.
- `glipizide-metformin`: excluded because its timing may be during or after the encounter.
- `glimepiride-pioglitazone`: excluded because its timing may be during or after the encounter.
- `metformin-rosiglitazone`: excluded because its timing may be during or after the encounter.
- `metformin-pioglitazone`: excluded because its timing may be during or after the encounter.
- `change`: excluded because its timing may be during or after the encounter.
- `diabetesMed`: excluded because its timing may be during or after the encounter.

## 7. Uncertain-timing features

- `weight`: excluded because availability at the prediction point is not documented.
- `medical_specialty`: excluded because availability at the prediction point is not documented.
- `diag_1`: excluded because availability at the prediction point is not documented.
- `diag_2`: excluded because availability at the prediction point is not documented.
- `diag_3`: excluded because availability at the prediction point is not documented.
- `number_diagnoses`: excluded because availability at the prediction point is not documented.

## 8. Final modeling feature set

- `race`
- `gender`
- `age`
- `admission_type_id`
- `admission_source_id`
- `payer_code`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`

## 9. Remaining limitations

- The selected prediction point is an academic design assumption, not a timestamp proven by the flat file.
- Prior utilization counts are treated as safe because UCI describes them as prior-year counts; the exact extraction process should still be recorded.
- Admission/early availability for payer and admission coding is plausible but not timestamp-verified.
- Diagnosis codes are excluded because coding timing is not established.
- No model was trained or evaluated in Step 21.

## Decision

FINAL MODELING FEATURE SET APPROVED: YES

The feature set is approved for the next modeling stage as a conservative research schema. Approval means the columns have passed this timing/leakage screen; it does not establish clinical validity or model performance.