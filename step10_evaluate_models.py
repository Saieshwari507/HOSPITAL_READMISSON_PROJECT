"""Step 10: evaluate trained models on the untouched test set."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "evaluation_outputs"
TARGET = "readmission_30day"
MODEL_NAMES = ["logistic_regression", "random_forest"]


def main() -> None:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}. Run Step 8 first.")

    test_data = pd.read_csv(TEST_PATH)
    features = test_data.drop(columns=[TARGET])
    target = test_data[TARGET]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    prediction_data = pd.DataFrame({TARGET: target.to_numpy()})

    print("=" * 70)
    print("STEP 10: MODEL EVALUATION")
    print("=" * 70)
    print(f"Test rows: {len(test_data)}")
    print(f"Positive test cases: {int(target.sum())}")

    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run Step 9 first.")

        model = joblib.load(model_path)
        probabilities = model.predict_proba(features)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        matrix = confusion_matrix(target, predictions, labels=[0, 1])

        metrics.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(target, probabilities),
                "average_precision": average_precision_score(target, probabilities),
                "brier_score": brier_score_loss(target, probabilities),
                "precision_at_0.5": precision_score(
                    target, predictions, zero_division=0
                ),
                "recall_at_0.5": recall_score(target, predictions, zero_division=0),
                "f1_at_0.5": f1_score(target, predictions, zero_division=0),
                "true_negatives": matrix[0, 0],
                "false_positives": matrix[0, 1],
                "false_negatives": matrix[1, 0],
                "true_positives": matrix[1, 1],
            }
        )
        prediction_data[f"{model_name}_probability"] = probabilities
        prediction_data[f"{model_name}_prediction_at_0.5"] = predictions

    metrics_data = pd.DataFrame(metrics).sort_values(
        "average_precision", ascending=False
    )
    metrics_data.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    prediction_data.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name in MODEL_NAMES:
        probabilities = prediction_data[f"{model_name}_probability"]
        thresholds = sorted(set(probabilities))
        precision_values = []
        recall_values = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            precision_values.append(precision_score(target, predictions, zero_division=0))
            recall_values.append(recall_score(target, predictions, zero_division=0))
        axes[0].plot(
            recall_values,
            precision_values,
            label=model_name.replace("_", " ").title(),
        )

    axes[0].set_title("Precision-Recall Curves")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metric_plot = metrics_data.set_index("model")[["roc_auc", "average_precision"]]
    metric_plot.plot(kind="bar", ax=axes[1], color=["#2f6f8f", "#c05a4e"])
    axes[1].set_title("Model Ranking")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=150)
    plt.close(figure)

    print("\nModel comparison:")
    print(metrics_data[["model", "roc_auc", "average_precision", "brier_score"]].round(4).to_string(index=False))
    print(f"\nEvaluation outputs saved to: {OUTPUT_DIR}")
    print("STEP 10 COMPLETE")


if __name__ == "__main__":
    main()
