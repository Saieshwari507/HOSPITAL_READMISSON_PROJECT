"""Step 25: global and local model explainability for the Step 24 final model."""

from pathlib import Path
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step22_preprocessing import APPROVED_FEATURES, TARGET, age_band_midpoint  # noqa: E402

GLOBAL_CSV = PROJECT_ROOT / "analysis" / "step25_global_importance.csv"
LOCAL_JSON = PROJECT_ROOT / "analysis" / "step25_local_explanation.json"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step25_explainability_report.md"
PLOT_PATH = PROJECT_ROOT / "analysis" / "step25_global_importance.png"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "final_model_config.joblib"
TRAIN_PROCESSED = PROJECT_ROOT / "data" / "train_processed.csv"
TOP_N = 15


def clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def global_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method = "model feature importances (tree-based)"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        method = "absolute linear coefficients on preprocessed features"
    else:
        raise ValueError("Unsupported model for explainability.")
    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    frame["feature_label"] = frame["feature"].map(clean_feature_name)
    return frame.sort_values("importance", ascending=False), method


def local_contributions(model, feature_names, instance_row):
    if hasattr(model, "coef_"):
        contributions = instance_row.to_numpy() * model.coef_[0]
        method = "linear coefficient times preprocessed feature value"
    elif hasattr(model, "feature_importances_"):
        centered = instance_row.to_numpy() - instance_row.mean()
        contributions = model.feature_importances_ * centered
        method = (
            "model-attributed directional signal using feature importances "
            "and deviation from the local row mean (approximate, not causal)"
        )
    else:
        raise ValueError("Unsupported model for local explainability.")
    frame = pd.DataFrame(
        {"feature": feature_names, "contribution": contributions}
    )
    frame["feature_label"] = frame["feature"].map(clean_feature_name)
    return frame.sort_values("contribution", ascending=False), method


def example_patient_raw() -> dict:
    return {
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": 75.0,
        "admission_type_id": 1,
        "admission_source_id": 7,
        "payer_code": "MC",
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
    }


def main() -> None:
    if not MODEL_PATH.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError("Run Step 24 before Step 25.")

    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    feature_names = list(config["feature_names"])
    train_processed = pd.read_csv(TRAIN_PROCESSED)
    train_x = train_processed.drop(columns=[TARGET])
    example_index = 0
    example_row = train_x.iloc[example_index]

    global_frame, global_method = global_importance(model, feature_names)
    global_frame.to_csv(GLOBAL_CSV, index=False)

    local_frame, local_method = local_contributions(model, feature_names, example_row)
    toward_higher = local_frame.head(5)
    toward_lower = local_frame.tail(5).sort_values("contribution")

    local_payload = {
        "example_patient_raw": example_patient_raw(),
        "estimated_probability": float(model.predict_proba(example_row.to_frame().T)[0, 1]),
        "local_method": local_method,
        "factors_toward_higher_estimated_risk": [
            {
                "feature": row["feature_label"],
                "contribution": float(row["contribution"]),
                "note": "Model-attributed factor associated with higher estimated risk.",
            }
            for _, row in toward_higher.iterrows()
        ],
        "factors_toward_lower_estimated_risk": [
            {
                "feature": row["feature_label"],
                "contribution": float(row["contribution"]),
                "note": "Model-attributed factor associated with lower estimated risk.",
            }
            for _, row in toward_lower.iterrows()
        ],
        "disclaimer": (
            "These are model-attributed factors associated with the prediction. "
            "They do not establish medical causation."
        ),
    }
    LOCAL_JSON.write_text(json.dumps(local_payload, indent=2), encoding="utf-8")

    top_plot = global_frame.head(TOP_N)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top_plot["feature_label"][::-1], top_plot["importance"][::-1], color="#2f6f8f")
    axis.set_title("Global model-attributed feature importance")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    report = [
        "# Step 25: Explainability",
        "",
        "## A. Global explanation",
        "",
        f"Method: {global_method}.",
        "The table ranks preprocessed features that contribute most overall to model predictions.",
        "",
        global_frame.head(TOP_N).round(6).to_string(index=False),
        "",
        "## B. Local explanation",
        "",
        f"Method: {local_method}.",
        "Example patient (raw approved inputs):",
        "",
        json.dumps(example_patient_raw(), indent=2),
        "",
        "Factors model-attributed toward higher estimated risk:",
        "",
    ]
    for item in local_payload["factors_toward_higher_estimated_risk"]:
        report.append(f"- `{item['feature']}` (contribution={item['contribution']:.6f})")
    report.extend(
        [
            "",
            "Factors model-attributed toward lower estimated risk:",
            "",
        ]
    )
    for item in local_payload["factors_toward_lower_estimated_risk"]:
        report.append(f"- `{item['feature']}` (contribution={item['contribution']:.6f})")
    report.extend(
        [
            "",
            "## Wording and limitations",
            "",
            "- Associations are model attributions, not proof that a feature causes readmission.",
            "- One-hot encoded categories represent patterns in historical data, not deterministic rules.",
            "- Local explanations for tree models use a simplified attribution approach suitable for academic demonstration.",
            "",
            "STEP 25 STATUS: COMPLETE",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("=" * 70)
    print("STEP 25: EXPLAINABILITY")
    print("=" * 70)
    print(f"Global method: {global_method}")
    print(f"Top feature: {global_frame.iloc[0]['feature_label']}")
    print("STEP 25 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
