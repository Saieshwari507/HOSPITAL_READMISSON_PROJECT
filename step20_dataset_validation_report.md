# Step 20: Dataset Validation Report

## Decision

**APPROVED WITH TIMING CAVEATS** for the final modeling stage after a documented, leakage-safe schema is created. No model was trained in Step 20.

## 1. Dataset overview

- Dataset: Diabetes 130-US Hospitals for Years 1999-2008.
- Source: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008.
- Rows/encounters: 101,766.
- Columns: 50.
- Population: hospitalized patient records with diabetes from 130 US hospitals/integrated delivery networks, 1999-2008, according to UCI documentation.

## 2. Dataset source

UCI documents the dataset, its variable table, the introductory paper, the source period, the inclusion criteria, and a CC BY 4.0 license. The project should cite UCI and the associated Strack et al. paper.

## 3. Target definition

- Source target: `readmitted`.
- Observed values: {'NO': 54864, '>30': 35545, '<30': 11357}.
- Derived final target: `readmission_30day = 1` when `readmitted == '<30'`; otherwise `0` for documented categories `NO` and `>30`.
- Derived positives: 11,357 (11.16%).
- UCI describes the task as early readmission within 30 days of discharge. The project must preserve the original three-level source target for auditability.

## 4. Prediction timing

- A precise timestamp column is **Not documented in the supplied flat file**.
- Admission type/source, demographics, prior utilization counts, and diagnoses are reasonable early candidates subject to the project’s timing definition.
- `discharge_disposition_id` and `time_in_hospital` are discharge/length-of-stay information and should be excluded from an admission-time model.
- Lab/medication counts, procedure counts, HbA1c/glucose result fields, medication change, and diabetes medication status may reflect information collected during the stay; exact availability time is **Not documented**.

## 5. Patient and encounter structure

- Patient ID: `patient_nbr` (71,518 unique patients).
- Encounter ID: `encounter_id` (101,766 unique encounters).
- Repeated patient rows: 30,248.
- Multiple encounters per patient are present, so patient-grouped splitting is required later.

## 6. Feature groups

- Demographics: race, gender, age, weight.
- Admission: admission type, admission source, payer, specialty.
- Prior utilization: outpatient, emergency, and inpatient visit counts.
- Diagnoses/comorbidity proxies: diagnosis codes and number of diagnoses.
- Clinical/treatment context: laboratory, procedure, medication, glucose, HbA1c, and diabetes-treatment fields.
- Full column and dtype inventory: `step20_validation_outputs/schema_summary.csv`.

## 7. Missingness

- Columns with missing values after treating `?` as missing: 7.
- `weight`: 98,569 missing (96.86%).
- `medical_specialty`: 49,949 missing (49.08%).
- `payer_code`: 40,256 missing (39.56%).
- `race`: 2,273 missing (2.23%).
- `diag_3`: 1,423 missing (1.40%).
- `diag_2`: 358 missing (0.35%).
- `diag_1`: 21 missing (0.02%).
- UCI documents that missing values exist; the meaning of each missingness pattern must still be considered during preprocessing.

## 8. Potential leakage

The following fields require exclusion or explicit timing justification for admission-time prediction:
- `discharge_disposition_id`.
- `time_in_hospital`.
- `num_lab_procedures`.
- `num_procedures`.
- `num_medications`.
- `max_glu_serum`.
- `A1Cresult`.
- `change`.
- `diabetesMed`.
- The target source column `readmitted` must never be used as a feature.
- `patient_nbr` and `encounter_id` are identifiers, not predictive features.

## 9. Dataset strengths

- Explicit documented readmission categories include `<30`.
- Large encounter count and substantially more positive cases than the current 269-case dataset.
- Separate patient and encounter identifiers with repeated encounters.
- Public source with variable documentation, inclusion criteria, paper, and license.
- Prior-utilization variables support a plausible early-risk design.

## 10. Dataset limitations

- Exact timestamps and feature availability times are Not documented in the flat file.
- The population is restricted to diabetic encounters and 130 hospitals from 1999-2008.
- The source target includes `>30` and `NO`; converting these to negative labels is a project decision that must be documented.
- Some variables describe care delivered during the hospitalization and may leak post-admission information.
- The dataset is not automatically representative of all hospital patients or current practice.

## 11. Candidate comparison

| Dataset | Encounters | Patients | 30-day target | IDs/repeats | Timing/documentation | Suitability |
| --- | ---: | ---: | --- | --- | --- | --- |
| Current `sepsis_icu_synthetic` | 5,000 | 5,000 | Binary column, provenance Not documented | Subject ID only; no repeats | Timing and encounter structure Not documented | Reject for final claims |
| UCI Diabetes 130-US Hospitals | 101,766 | 71,518 | `<30`, `>30`, `NO`; explicit early-readmission task | Patient and encounter IDs; repeats present | Strong source/feature documentation; exact timestamps Not documented | Recommended with timing caveats |
| MIMIC-IV | 546,028 hospitalizations; 223,452 hospitalized individuals | 223,452 hospitalized individuals | Not supplied as a ready-made 30-day target; must be derived | Patient/hospitalization/ICU IDs; repeats present | Excellent timestamps and provenance; credentialed access required | Strong future candidate, larger setup burden |

## 12. Final recommendation

Select the UCI Diabetes 130-US Hospitals dataset for this project’s replacement schema. It is better documented and structurally defensible than the current synthetic workbook, but the final model must use an admission/early feature subset and preserve the target transformation audit trail.

## Remaining uncertainties

- Exact timing for several lab, medication, procedure, and result fields is **Not documented**.
- Exact planned-readmission and incomplete-follow-up handling is **Not documented** in the dataset file.
- The derived binary target is a project transformation from UCI’s three-level target and must be recorded in the final report.