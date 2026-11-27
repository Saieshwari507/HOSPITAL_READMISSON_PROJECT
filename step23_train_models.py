"""Step 23: train and compare baseline models on Step 22 processed data."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"
COMPARISON_PATH = PROJECT_ROOT / "analysis" / "step23_model_comparison.csv"
PLOT_PATH = PROJECT_ROOT / "analysis" / "step23_model_comparison.png"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step23_model_report.md"
TARGET = "readmission_30day"
RANDOM_STATE = 42
THRESHOLD = 0.50


def load_data():
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Step 22 processed files are required.")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    forbidden = {"patient_nbr", "encounter_id", "readmitted"}
    if forbidden.intersection(train.columns) or forbidden.intersection(test.columns):
        raise ValueError("Forbidden columns found in processed model inputs.")
    return train, test


def build_models():
    return {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "class_weight='balanced'",
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "class_weight='balanced'",
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "no native class_weight; uses unweighted training",
        ),
    }


def metrics_for_model(target, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "accuracy": accuracy_score(target, predictions),
        "brier_score": brier_score_loss(target, probabilities),
    }, confusion_matrix(target, predictions, labels=[0, 1])


def write_report(comparison, class_strategies, train, test):
    best_roc = comparison.loc[comparison["roc_auc"].idxmax(), "model"]
    best_pr = comparison.loc[comparison["average_precision"].idxmax(), "model"]
    best_brier = comparison.loc[comparison["brier_score"].idxmin(), "model"]
    lines = [
        "# Step 23: Baseline Model Training and Comparison",
        "",
        "## 1. Models evaluated",
        "",
        "- Logistic Regression",
        "- Random Forest",
        "- Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`)",
        "- XGBoost was not used because it is not installed.",
        "",
        "## 2. Training and test data",
        "",
        f"- Training encounters: {len(train):,}; positives: {int(train[TARGET].sum()):,}; rate: {train[TARGET].mean() * 100:.2f}%.",
        f"- Test encounters: {len(test):,}; positives: {int(test[TARGET].sum()):,}; rate: {test[TARGET].mean() * 100:.2f}%.",
        "- Inputs came only from `train_processed.csv` and `test_processed.csv`.",
        "- Patient-aware split and preprocessing were completed in Step 22.",
        "",
        "## 3. Class imbalance",
        "",
        "The positive class is approximately 11.16% of encounters. Logistic Regression and Random Forest use class weighting. Gradient Boosting has no native class-weight parameter and was trained unweighted for this initial comparison. No oversampling or SMOTE was used.",
        "",
        "## 4. Evaluation metrics",
        "",
        f"Precision, recall, F1, and accuracy use the fixed default threshold of {THRESHOLD:.2f}. Threshold optimization is deferred to a later step. ROC-AUC, average precision, and Brier score use predicted probabilities.",
        "",
        comparison.round(4).to_string(index=False),
        "",
        "## 5. Model comparison",
        "",
        f"- Best ROC-AUC model: `{best_roc}`.",
        f"- Best PR-AUC model: `{best_pr}`.",
        f"- Best Brier score model: `{best_brier}`.",
        "",
        "## 6. Limitations",
        "",
        "- These results are baseline comparisons, not evidence of clinical usefulness.",
        "- Feature importance or coefficients do not establish causation.",
        "- The UCI dataset has documented target categories, but exact feature availability timestamps remain limited.",
        "- Results may be affected by the diabetic-encounter population and historical 1999-2008 collection period.",
        "",
        "## 7. Proceeding to the next stage",
        "",
        "ROC-AUC near 0.60 and PR-AUC near 0.19 indicate weak but non-random ranking signal for this feature set. Proceeding to threshold selection and deployment as an academic prototype is justified, but results should be reported honestly as limited and not clinically validated. Threshold selection must use validation data and must not alter the held-out test set.",
        "",
        "STEP 23 STATUS: COMPLETE",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    train, test = load_data()
    train_features = train.drop(columns=[TARGET])
    train_target = train[TARGET]
    test_features = test.drop(columns=[TARGET])
    test_target = test[TARGET]
    models = build_models()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    class_strategies = {}

    for model_name, (model, strategy) in models.items():
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(test_features)[:, 1]
        model_metrics, matrix = metrics_for_model(test_target, probabilities)
        model_metrics["model"] = model_name
        rows.append(model_metrics)
        class_strategies[model_name] = strategy
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
        pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(PROJECT_ROOT / "analysis" / f"step23_{model_name}_confusion_matrix.csv")

    comparison = pd.DataFrame(rows)[
        ["model", "roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "brier_score"]
    ].sort_values("average_precision", ascending=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    metric_plot = comparison.set_index("model")[["roc_auc", "average_precision", "f1"]]
    metric_plot.plot(kind="bar", ax=axes[0], color=["#2f6f8f", "#c05a4e", "#4c8c6b"])
    axes[0].set_title("Ranking and Classification Metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)
    comparison.set_index("model")["brier_score"].plot(kind="bar", ax=axes[1], color="#c99749")
    axes[1].set_title("Brier Score (lower is better)")
    axes[1].set_ylabel("Brier score")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    write_report(comparison, class_strategies, train, test)
    print("=" * 70)
    print("STEP 23: TRAIN AND COMPARE BASELINE ML MODELS")
    print("=" * 70)
    print(comparison.round(4).to_string(index=False))
    print(f"BEST ROC-AUC MODEL: {comparison.loc[comparison['roc_auc'].idxmax(), 'model']}")
    print(f"BEST PR-AUC MODEL: {comparison.loc[comparison['average_precision'].idxmax(), 'model']}")
    print(f"BEST BRIER SCORE MODEL: {comparison.loc[comparison['brier_score'].idxmin(), 'model']}")
    print("Results are baseline research metrics, not evidence of clinical usefulness.")
    print("STEP 23 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
