# Step 22: Patient-Aware Splitting and Preprocessing

## 1. Dataset overview

- Source rows: 101,766 encounters.
- Source patients: 71,518.
- Source target: `readmitted == '<30'`.

## 2. Prediction target

`readmission_30day = 1` when `readmitted == '<30'`; otherwise `0`.

## 3. Approved features

- `race`
- `gender`
- `age`
- `admission_type_id`
- `admission_source_id`
- `payer_code`
- `number_outpatient`
- `number_emergency`
- `number_inpatient`

## 4. Patient-level splitting method

A reproducible `GroupShuffleSplit` search was used with `patient_nbr` as the grouping variable only. Candidate grouped splits were compared for approximate 80/20 patient allocation and target-rate similarity. `patient_nbr` was never passed to the preprocessing transformer.

## 5. Split summary

- Train patients: 57,214; encounters: 81,484; positives: 9,094; rate: 11.16%.
- Test patients: 14,304; encounters: 20,282; positives: 2,263; rate: 11.16%.
- Patient overlap: 0.

## 6. Missing-value handling

- Numeric features use median imputation fitted on training rows only.
- Categorical features use a dedicated `Missing` category fitted on training rows only.
- Test values are transformed using the already-fitted training transformer.

## 7. Categorical encoding

One-hot encoding is used for `race`, `gender`, `payer_code`, `admission_type_id`, and `admission_source_id`. Unknown test categories are ignored safely.

## 8. Numerical preprocessing

`age`, `number_outpatient`, `number_emergency`, and `number_inpatient` use training-fitted median imputation and standard scaling. One-hot categorical columns are not scaled separately.
- UCI stores `age` as grouped labels; each band is converted to its midpoint before the training-only numeric imputer/scaler.

## 9. Leakage prevention

- `patient_nbr` is used only for grouping and is absent from processed features.
- `encounter_id` is absent from processed features.
- `readmitted` is absent from processed features.
- The target is retained only as `readmission_30day` in the processed output for later supervised modeling.
- No model was trained and no ROC-AUC was calculated.

## 10. Final preprocessing pipeline

Patient-aware split -> fit numeric median imputation and scaling on train -> fit categorical Missing handling and one-hot encoding on train -> transform train and test with the same fitted transformer.

## Verification

PATIENT OVERLAP: 0
PREPROCESSING LEAKAGE CHECK: PASSED

STEP 22 STATUS: COMPLETE