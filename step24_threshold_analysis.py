"""Step 24: select final model and optimize decision threshold on validation data only."""

from pathlib import Path
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step22_preprocessing import (  # noqa: E402
    APPROVED_FEATURES,
    GROUP,
    RANDOM_STATE,
    TARGET,
    build_preprocessor,
    choose_group_split,
    load_data,
)
from step23_train_models import build_models  # noqa: E402

TRAIN_PROCESSED = PROJECT_ROOT / "data" / "train_processed.csv"
TEST_PROCESSED = PROJECT_ROOT / "data" / "test_processed.csv"
COMPARISON_PATH = PROJECT_ROOT / "analysis" / "step23_model_comparison.csv"
THRESHOLD_CSV = PROJECT_ROOT / "analysis" / "step24_threshold_results.csv"
THRESHOLD_PLOT = PROJECT_ROOT / "analysis" / "step24_threshold_analysis.png"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step24_final_model_report.md"
MODEL_DIR = PROJECT_ROOT / "models"
SELECTION_PATH = PROJECT_ROOT / "analysis" / "step24_final_selection.json"

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
VAL_SIZE = 0.15
MIN_RECALL_FOR_SELECTION = 0.20


def select_model_name(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        ["average_precision", "roc_auc", "brier_score"],
        ascending=[False, False, True],
    )
    return str(ranked.iloc[0]["model"])


def build_model_by_name(model_name: str):
    model, _ = build_models()[model_name]
    return model


def fit_with_imbalance(model, features, target, model_name: str):
    """Match Step 23: class-weighted linear/forest models; unweighted gradient boosting."""
    if model_name == "gradient_boosting":
        model.fit(features, target)
    else:
        model.fit(features, target)


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(results: pd.DataFrame) -> float:
    eligible = results[results["recall"] >= MIN_RECALL_FOR_SELECTION]
    if not eligible.empty:
        return float(eligible.sort_values(["f1", "precision"], ascending=False).iloc[0]["threshold"])
    return float(results.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"])


def main() -> None:
    if not COMPARISON_PATH.is_file():
        raise FileNotFoundError("Run Step 23 before Step 24.")

    comparison = pd.read_csv(COMPARISON_PATH)
    model_name = select_model_name(comparison)
    data = load_data()
    train_idx, _ = choose_group_split(data)
    train_raw = data.iloc[train_idx].copy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    sub_train_idx, val_idx = next(
        splitter.split(train_raw, train_raw[TARGET], train_raw[GROUP])
    )
    sub_train = train_raw.iloc[sub_train_idx]
    val_raw = train_raw.iloc[val_idx]

    preprocessor = build_preprocessor()
    preprocessor.fit(sub_train[APPROVED_FEATURES])
    sub_x = preprocessor.transform(sub_train[APPROVED_FEATURES])
    val_x = preprocessor.transform(val_raw[APPROVED_FEATURES])
    sub_y = sub_train[TARGET]
    val_y = val_raw[TARGET]

    feature_names = list(preprocessor.get_feature_names_out())
    sub_frame = pd.DataFrame(sub_x, columns=feature_names)
    val_frame = pd.DataFrame(val_x, columns=feature_names)

    model = build_model_by_name(model_name)
    fit_with_imbalance(model, sub_frame, sub_y, model_name)
    val_prob = model.predict_proba(val_frame)[:, 1]

    threshold_rows = [threshold_metrics(val_y, val_prob, t) for t in THRESHOLDS]
    threshold_df = pd.DataFrame(threshold_rows)
    selected_threshold = choose_threshold(threshold_df)
    threshold_df["selected"] = threshold_df["threshold"] == selected_threshold
    threshold_df.to_csv(THRESHOLD_CSV, index=False)

    train_processed = pd.read_csv(TRAIN_PROCESSED)
    test_processed = pd.read_csv(TEST_PROCESSED)
    train_x = train_processed.drop(columns=[TARGET])
    train_y = train_processed[TARGET]
    test_x = test_processed.drop(columns=[TARGET])
    test_y = test_processed[TARGET]

    preprocessor_full = build_preprocessor()
    preprocessor_full.fit(train_raw[APPROVED_FEATURES])
    final_model = build_model_by_name(model_name)
    fit_with_imbalance(final_model, train_x, train_y, model_name)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "final_model.joblib")
    joblib.dump(preprocessor_full, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(
        {
            "model_name": model_name,
            "threshold": selected_threshold,
            "feature_names": feature_names,
        },
        MODEL_DIR / "final_model_config.joblib",
    )

    test_prob = final_model.predict_proba(test_x)[:, 1]
    test_pred = (test_prob >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred, labels=[0, 1]).ravel()

    test_metrics = {
        "roc_auc": roc_auc_score(test_y, test_prob),
        "average_precision": average_precision_score(test_y, test_prob),
        "precision": precision_score(test_y, test_pred, zero_division=0),
        "recall": recall_score(test_y, test_pred, zero_division=0),
        "f1": f1_score(test_y, test_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "brier_score": brier_score_loss(test_y, test_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    val_selected = threshold_df[threshold_df["threshold"] == selected_threshold].iloc[0]
    selection = {
        "selected_model": model_name,
        "selected_threshold": selected_threshold,
        "validation_metrics_at_selected_threshold": val_selected.to_dict(),
        "test_metrics": test_metrics,
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    axes[0].plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    axes[0].plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    axes[0].axvline(selected_threshold, color="#c05a4e", linestyle="--", label="Selected")
    axes[0].set_title("Validation threshold trade-off")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["#4c8c6b", "#c99749", "#2f6f8f", "#c05a4e"])
    axes[1].set_title("Final test confusion matrix")
    axes[1].set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(THRESHOLD_PLOT, dpi=150)
    plt.close(figure)

    report_lines = [
        "# Step 24: Final Model and Threshold Selection",
        "",
        "## Selected final model",
        "",
        f"- Model: `{model_name}`.",
        f"- Selection priority: highest validation-stage PR-AUC / average precision from Step 23, then ROC-AUC, then Brier score.",
        "",
        "## Selected threshold",
        "",
        f"- Threshold: `{selected_threshold:.2f}`.",
        f"- Thresholds evaluated on validation only: {', '.join(f'{t:.2f}' for t in THRESHOLDS)}.",
        f"- Rule: among thresholds with recall >= {MIN_RECALL_FOR_SELECTION:.2f}, choose the highest F1; if none qualify, choose the best F1 overall.",
        "- This avoids selecting a threshold only because it maximizes recall.",
        "",
        "## Validation results (threshold selection)",
        "",
        threshold_df.round(4).to_string(index=False),
        "",
        f"At the selected threshold on validation: precision={val_selected['precision']:.4f}, recall={val_selected['recall']:.4f}, F1={val_selected['f1']:.4f}, specificity={val_selected['specificity']:.4f}, sensitivity={val_selected['sensitivity']:.4f}.",
        "",
        "## Final held-out test evaluation (untouched during threshold selection)",
        "",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- PR-AUC / average precision: {test_metrics['average_precision']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F1: {test_metrics['f1']:.4f}",
        f"- Specificity: {test_metrics['specificity']:.4f}",
        f"- Sensitivity: {test_metrics['sensitivity']:.4f}",
        f"- Brier score: {test_metrics['brier_score']:.4f}",
        f"- Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "",
        "## Limitations",
        "",
        "- Performance remains modest because only admission-time features approved in Step 21 were used.",
        "- Threshold selection used a patient-grouped validation subset of training data; it did not use the held-out test set.",
        "- Results describe estimated readmission risk patterns in historical UCI data, not clinical causation.",
        "",
        "STEP 24 STATUS: COMPLETE",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 70)
    print("STEP 24: FINAL MODEL AND THRESHOLD SELECTION")
    print("=" * 70)
    print(f"Selected model: {model_name}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print("Final test ROC-AUC:", round(test_metrics["roc_auc"], 4))
    print("Final test PR-AUC:", round(test_metrics["average_precision"], 4))
    print("STEP 24 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
