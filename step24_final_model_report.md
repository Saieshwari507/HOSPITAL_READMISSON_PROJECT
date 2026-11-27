# Step 24: Final Model and Threshold Selection

## Selected final model

- Model: `gradient_boosting`.
- Selection priority: highest validation-stage PR-AUC / average precision from Step 23, then ROC-AUC, then Brier score.

## Selected threshold

- Threshold: `0.15`.
- Thresholds evaluated on validation only: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50.
- Rule: among thresholds with recall >= 0.20, choose the highest F1; if none qualify, choose the best F1 overall.
- This avoids selecting a threshold only because it maximizes recall.

## Validation results (threshold selection)

 threshold  precision  recall     f1  accuracy  specificity  sensitivity  tp   fp    tn   fn  selected
      0.10     0.1499  0.6140 0.2409    0.5825       0.5787       0.6140 797 4521  6209  501     False
      0.15     0.2104  0.2827 0.2413    0.8081       0.8717       0.2827 367 1377  9353  931      True
      0.20     0.2822  0.1333 0.1811    0.8699       0.9590       0.1333 173  440 10290 1125     False
      0.25     0.3430  0.0732 0.1206    0.8849       0.9830       0.0732  95  182 10548 1203     False
      0.30     0.3723  0.0539 0.0942    0.8881       0.9890       0.0539  70  118 10612 1228     False
      0.35     0.4653  0.0362 0.0672    0.8915       0.9950       0.0362  47   54 10676 1251     False
      0.40     0.5862  0.0262 0.0501    0.8929       0.9978       0.0262  34   24 10706 1264     False
      0.50     0.5000  0.0046 0.0092    0.8921       0.9994       0.0046   6    6 10724 1292     False

At the selected threshold on validation: precision=0.2104, recall=0.2827, F1=0.2413, specificity=0.8717, sensitivity=0.2827.

## Final held-out test evaluation (untouched during threshold selection)

- ROC-AUC: 0.6188
- PR-AUC / average precision: 0.1977
- Precision: 0.2145
- Recall: 0.2762
- F1: 0.2415
- Specificity: 0.8730
- Sensitivity: 0.2762
- Brier score: 0.0962
- Confusion matrix: TN=15730, FP=2289, FN=1638, TP=625

## Limitations

- Performance remains modest because only admission-time features approved in Step 21 were used.
- Threshold selection used a patient-grouped validation subset of training data; it did not use the held-out test set.
- Results describe estimated readmission risk patterns in historical UCI data, not clinical causation.

STEP 24 STATUS: COMPLETE