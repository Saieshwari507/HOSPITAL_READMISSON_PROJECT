# Step 6: Historical Readmission Patterns

Overall readmission rate: 5.38%.
Only groups with at least 30 patients are included.

## Highest observed subgroup rates

| Variable | Group | Patients | Readmission rate | Difference from baseline |
| --- | --- | ---: | ---: | ---: |
| sepsis_label | 1 | 750 | 7.60% | +2.22 percentage points |
| sofa_score | [9.0, 17.0] | 851 | 7.52% | +2.14 percentage points |
| age | [65.0, 74.0] | 1154 | 7.02% | +1.64 percentage points |
| apache_iv | [43.525, 101.2] | 1250 | 6.80% | +1.42 percentage points |
| creatinine | [2.64, 9.54] | 1110 | 6.76% | +1.30 percentage points |
| qsofa | [1.0, 3.0] | 918 | 6.64% | +1.26 percentage points |
| hospital_admit_source | Transfer | 490 | 6.53% | +1.15 percentage points |
| sirs_criteria | [1.0, 4.0] | 1145 | 6.46% | +1.08 percentage points |
| insurance | Medicare | 1912 | 6.43% | +1.05 percentage points |
| hemoglobin | [11.7, 13.06] | 1116 | 6.18% | +0.70 percentage points |

These are historical associations, not causal conclusions. They must be confirmed with a leakage-safe train/test evaluation.