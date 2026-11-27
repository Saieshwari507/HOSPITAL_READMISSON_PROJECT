# Step 18: Prediction Timing and Target Provenance Audit

## Scope and evidence

- Original dataset: `data\ML dataset.xlsx`.
- Rows and columns: 5000 x 77.
- Existing feature-engineered data and project scripts were inspected.
- The held-out test set was not loaded or used.
- Verified facts, observations, hypotheses, and unknowns are explicitly separated below.

## Verified facts

- Target values observed: {0: 4731, 1: 269}.
- Target missing values: 0.
- subject_id duplicate rows: 0.
- Timestamp-like columns present: icu_admit_time_hour.
- Explicit discharge/readmission date columns present: none.
- Timing classifications generated: {'timing unknown': 46, 'likely available at hospital admission': 20, 'potentially post-treatment/post-outcome': 9, 'identifier': 1, 'post-outcome / target': 1}.

## Feature-group coverage

- **Demographics**: 7 variables inspected.
- **Comorbidities**: 10 variables inspected.
- **Vital signs**: 25 variables inspected.
- **Laboratory**: 15 variables inspected.
- **Treatment / ICU**: 8 variables inspected.
- **Severity scores**: 5 variables inspected.
- **Hospital / encounter context**: 4 variables inspected.
- **Engineered features**: 0 variables inspected.

## Observations

- The workbook contains `icu_admit_time_hour` and `day_of_week`, but not a full admission timestamp.
- The workbook contains one row per unique `subject_id`; no repeated encounters can be verified from this file.
- The target has observed associations with sepsis, ICU length of stay, treatment, and severity variables, but these are descriptive relationships only.
- Row-order and subject-ID quartile rates were measured because synthetic or sorted data can create apparent patterns; neither variable is documented as meaningful.

## Hypotheses requiring confirmation

- Demographics, comorbidities, admission source, admission hour, and day of week are plausibly available at or near admission, but this is not established by timestamps.
- Vital signs, laboratory values, severity scores, and `sepsis_label` may be available shortly after admission or later; their collection/calculation windows are undocumented.
- Treatment variables and `icu_los_hours` may encode information that becomes available only after treatment or clinical progression.

## Unknown information

- The project files do not explain how `readmission_30day` was generated.
- There are no discharge timestamps, readmission timestamps, encounter IDs, or explicit encounter windows in the workbook.
- It cannot be determined whether labels are independently generated, derived from an unavailable source, or based on a synthetic rule.

## Target provenance evidence

The provenance summary CSV contains the observed target relationships. Those relationships do not prove that any feature generated the target, and no causal interpretation is made.

## Answers

### A. Most defensible prediction point
The most defensible point is a conservative admission or immediate-intake prediction using only demographics, known comorbidities, admission source, calendar context, and variables whose collection time is explicitly confirmed. The current dataset does not fully verify even this point, so it should be treated as a provisional research definition.

### B. Variables to exclude pending timing evidence
Exclude `icu_los_hours`, treatment/support variables, and any laboratory, severity, or sepsis variable whose timestamp is not confirmed for the intended prediction point. Do not use the target or `subject_id`.

### C. Variables safe to consider for an early model
Demographics, documented comorbidities, `hospital_admit_source`, `icu_admit_time_hour`, and `day_of_week` are the safest candidates conceptually, subject to confirmation from the data owner. Their safety is a timing hypothesis, not a verified fact.

### D. Can target provenance be verified?
No. The project files verify the target column and its observed values, but they do not verify how the labels were generated.

### E. Is the dataset suitable for a meaningful model?
The dataset is suitable for a documented research experiment, but its suitability for a meaningful readmission model is unproven. Missing provenance, missing encounter timestamps, one row per unique subject, and weak Step 17 signal substantially limit confidence.

### F. Gradient boosting or dataset investigation?
Investigate and, if possible, replace or augment the dataset first. Do not proceed directly to gradient boosting as though it will solve the provenance and timing uncertainty. A later conservative gradient-boosting comparison is reasonable only after target provenance and prediction timing are clarified.