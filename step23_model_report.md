# Step 23: Baseline Model Training and Comparison

## 1. Models evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`)
- XGBoost was not used because it is not installed.

## 2. Training and test data

- Training encounters: 81,484; positives: 9,094; rate: 11.16%.
- Test encounters: 20,282; positives: 2,263; rate: 11.16%.
- Inputs came only from `train_processed.csv` and `test_processed.csv`.
- Patient-aware split and preprocessing were completed in Step 22.

## 3. Class imbalance

The positive class is approximately 11.16% of encounters. Logistic Regression and Random Forest use class weighting. Gradient Boosting has no native class-weight parameter and was trained unweighted for this initial comparison. No oversampling or SMOTE was used.

## 4. Evaluation metrics

Precision, recall, F1, and accuracy use the fixed default threshold of 0.50. Threshold optimization is deferred to a later step. ROC-AUC, average precision, and Brier score use predicted probabilities.

              model  roc_auc  average_precision  precision  recall     f1  accuracy  brier_score
  gradient_boosting   0.6188             0.1977     0.7778  0.0031 0.0062    0.8887       0.0962
logistic_regression   0.6157             0.1930     0.1695  0.4375 0.2444    0.6982       0.2331
      random_forest   0.5965             0.1714     0.1679  0.4110 0.2384    0.7071       0.2104

## 5. Model comparison

- Best ROC-AUC model: `gradient_boosting`.
- Best PR-AUC model: `gradient_boosting`.
- Best Brier score model: `gradient_boosting`.

## 6. Limitations

- These results are baseline comparisons, not evidence of clinical usefulness.
- Feature importance or coefficients do not establish causation.
- The UCI dataset has documented target categories, but exact feature availability timestamps remain limited.
- Results may be affected by the diabetic-encounter population and historical 1999-2008 collection period.

## 7. Proceeding to the next stage

ROC-AUC near 0.60 and PR-AUC near 0.19 indicate weak but non-random ranking signal for this feature set. Proceeding to threshold selection and deployment as an academic prototype is justified, but results should be reported honestly as limited and not clinically validated. Threshold selection must use validation data and must not alter the held-out test set.

STEP 23 STATUS: COMPLETE