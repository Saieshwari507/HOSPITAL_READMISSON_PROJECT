# Step 17: Signal Validation

## Design

- Training rows: 4000.
- Training positive cases: 215.
- Training prevalence: 5.38%.
- Cross-validation: 5-fold stratified CV repeated 3 times (15 validation folds).
- The existing held-out test set was not loaded or used by this experiment.
- Baseline model thresholds use 0.5 only for descriptive classification metrics; no threshold was selected.

## Results

Metrics are reported as mean +/- standard deviation across validation folds.

| Feature group | Model | ROC-AUC | PR-AUC | Brier | Recall | Precision | F1 | Specificity | Predicted-positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| admission_early_clinical | unweighted_logistic_regression | 0.517 +/- 0.032 | 0.068 +/- 0.014 | 0.052 +/- 0.001 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| admission_early_clinical | class_balanced_logistic_regression | 0.509 +/- 0.040 | 0.065 +/- 0.012 | 0.232 +/- 0.007 | 0.386 +/- 0.073 | 0.057 +/- 0.011 | 0.099 +/- 0.018 | 0.635 +/- 0.019 | 0.366 +/- 0.018 |
| admission_early_clinical | majority_class_classifier | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.054 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| admission_early_clinical | prevalence_only_probability | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_only | class_balanced_logistic_regression | 0.541 +/- 0.048 | 0.063 +/- 0.008 | 0.244 +/- 0.004 | 0.495 +/- 0.088 | 0.060 +/- 0.010 | 0.107 +/- 0.018 | 0.561 +/- 0.016 | 0.442 +/- 0.016 |
| demographics_only | unweighted_logistic_regression | 0.540 +/- 0.047 | 0.063 +/- 0.008 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_only | majority_class_classifier | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.054 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_only | prevalence_only_probability | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_plus_comorbidities | unweighted_logistic_regression | 0.537 +/- 0.040 | 0.064 +/- 0.010 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_plus_comorbidities | class_balanced_logistic_regression | 0.536 +/- 0.040 | 0.065 +/- 0.011 | 0.240 +/- 0.005 | 0.456 +/- 0.080 | 0.061 +/- 0.011 | 0.107 +/- 0.019 | 0.599 +/- 0.017 | 0.404 +/- 0.015 |
| demographics_plus_comorbidities | majority_class_classifier | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.054 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| demographics_plus_comorbidities | prevalence_only_probability | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| full_current_feature_set | unweighted_logistic_regression | 0.507 +/- 0.032 | 0.065 +/- 0.013 | 0.053 +/- 0.001 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| full_current_feature_set | class_balanced_logistic_regression | 0.500 +/- 0.036 | 0.064 +/- 0.013 | 0.233 +/- 0.007 | 0.371 +/- 0.066 | 0.054 +/- 0.009 | 0.095 +/- 0.016 | 0.633 +/- 0.016 | 0.367 +/- 0.016 |
| full_current_feature_set | majority_class_classifier | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.054 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |
| full_current_feature_set | prevalence_only_probability | 0.500 +/- 0.000 | 0.054 +/- 0.000 | 0.051 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 |

## Timing and leakage review

- `icu_los_hours` is likely known only after the encounter has progressed and may be unavailable at admission-time prediction.
- Treatment variables such as vasopressors, ventilation, antibiotics, fluids, sedation, and insulin may occur after clinical deterioration.
- Severity scores, laboratory values, and `sepsis_label` require an explicit timestamp definition before use.
- The admission/early clinical group is a statistical experiment, not proof that these variables are available at the intended prediction point.
- No feature is interpreted as causing readmission; results are associations and model performance measurements only.
- Variables present in the current data requiring timing review: icu_los_hours, vasopressors_flag, fio2_percent, antibiotics_24h, fluids_ml_24h, sedation_score, vasopressor_dose_mcg_kg_min, insulin_infusion_flag, treatment_flag_count.

## Interpretation framework

### A. Which baseline performs best?
Unweighted logistic regression is the strongest probability baseline overall. The best ROC-AUC is 0.541 +/- 0.048 for demographics_only; the best PR-AUC is 0.068 +/- 0.014 for admission_early_clinical. Class-balanced logistic regression improves recall at threshold 0.5 but has poor Brier scores.

### B. Is performance consistently above prevalence/random baselines?
No. Some mean scores exceed the baselines (prevalence PR-AUC 0.054; random ROC-AUC 0.500), but the gains are small and the fold standard deviations are large. This is not consistent evidence of useful generalization (some mean metrics are above baseline).

### C. Which feature group contains the most useful signal?
The strongest group by mean PR-AUC is `admission_early_clinical`. Demographics-only has the strongest mean ROC-AUC, while the admission/early-clinical group has the strongest mean PR-AUC. Neither group shows strong or stable predictive performance.

### D. Is there enough predictive information?
No, not for reliable individual prediction at this stage. The results are close to random ranking and only modestly above the prevalence PR baseline. The dataset may contain weak association, but it does not yet demonstrate enough stable predictive information.

### E. What should we do next?
First confirm target provenance and prediction-time timestamps, especially for post-admission and treatment variables. Then compare the strongest simple baseline with one conservatively regularized gradient-boosted tabular model using repeated training-only cross-validation. Keep the held-out test set untouched until the final model and threshold are locked.