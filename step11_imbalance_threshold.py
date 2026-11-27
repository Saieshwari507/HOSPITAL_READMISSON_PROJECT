"""Step 11: address class imbalance and select operating thresholds."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "threshold_outputs"
TARGET = "readmission_30day"
MODEL_NAMES = ["logistic_regression", "random_forest"]
RANDOM_STATE = 42
BETA = 2


def score_at_threshold(target: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    true_negatives = int(((target == 0) & (predictions == 0)).sum())
    false_positives = int(((target == 0) & (predictions == 1)).sum())
    specificity = true_negatives / (true_negatives + false_positives)
    return {
        "threshold": threshold,
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "f2": fbeta_score(target, predictions, beta=BETA, zero_division=0),
        "specificity": specificity,
        "accuracy": accuracy_score(target, predictions),
        "predicted_positive_percentage": predictions.mean() * 100,
    }


def choose_threshold(target: pd.Series, probabilities: np.ndarray) -> tuple:
    candidate_thresholds = np.arange(0.01, 1.00, 0.01)
    results = pd.DataFrame(
        [score_at_threshold(target, probabilities, threshold) for threshold in candidate_thresholds]
    )
    selected = results.sort_values(
        ["f2", "recall", "precision"], ascending=False
    ).iloc[0]
    return float(selected["threshold"]), results


def main() -> None:
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Train and test data are required. Run Step 8 first.")

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    train_features = train_data.drop(columns=[TARGET])
    train_target = train_data[TARGET]
    test_features = test_data.drop(columns=[TARGET])
    test_target = test_data[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    test_rows = []
    plot_data = []

    print("=" * 70)
    print("STEP 11: CLASS IMBALANCE AND THRESHOLD SELECTION")
    print("=" * 70)
    print("Threshold selection uses out-of-fold training predictions only.")

    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run Step 9 first.")

        fitted_model = joblib.load(model_path)
        out_of_fold_probabilities = cross_val_predict(
            clone(fitted_model),
            train_features,
            train_target,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold, threshold_results = choose_threshold(
            train_target, out_of_fold_probabilities
        )
        threshold_results.insert(0, "model", model_name)
        threshold_results.to_csv(
            OUTPUT_DIR / f"{model_name}_threshold_search.csv", index=False
        )

        test_probabilities = fitted_model.predict_proba(test_features)[:, 1]
        test_scores = score_at_threshold(test_target, test_probabilities, threshold)
        test_scores["model"] = model_name
        test_rows.append(test_scores)
        selected_rows.append(
            {
                "model": model_name,
                "selected_threshold": threshold,
                "training_oof_f2": float(
                    threshold_results.loc[
                        threshold_results["threshold"] == threshold, "f2"
                    ].iloc[0]
                ),
            }
        )

        for _, row in threshold_results.iterrows():
            plot_data.append(row)

    selected_data = pd.DataFrame(selected_rows)
    test_data_summary = pd.DataFrame(test_rows)
    selected_data.to_csv(OUTPUT_DIR / "selected_thresholds.csv", index=False)
    test_data_summary.to_csv(OUTPUT_DIR / "test_metrics_at_selected_threshold.csv", index=False)

    plot_frame = pd.DataFrame(plot_data)
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name in MODEL_NAMES:
        model_rows = plot_frame[plot_frame["model"] == model_name]
        axis.plot(model_rows["threshold"], model_rows["f2"], label=model_name.replace("_", " ").title())
    axis.set_title("Training Out-of-Fold F2 by Decision Threshold")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("F2 score")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "threshold_selection.png", dpi=150)
    plt.close(figure)

    print("\nSelected thresholds:")
    print(selected_data.round(4).to_string(index=False))
    print("\nTest performance at selected thresholds:")
    print(test_data_summary.round(4).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("STEP 11 COMPLETE")


if __name__ == "__main__":
    main()
