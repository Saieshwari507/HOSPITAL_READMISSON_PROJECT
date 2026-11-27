"""Step 12: extract global and local model explanations."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "explanation_outputs"
TARGET = "readmission_30day"
TOP_N = 20


def get_feature_names(model_pipeline) -> list:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def save_logistic_explanation(model_pipeline, test_features: pd.DataFrame) -> None:
    feature_names = get_feature_names(model_pipeline)
    classifier = model_pipeline.named_steps["model"]
    coefficients = classifier.coef_[0]
    global_data = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
            "direction": [
                "increases estimated risk" if coefficient > 0 else "decreases estimated risk"
                for coefficient in coefficients
            ],
        }
    ).sort_values("absolute_coefficient", ascending=False)
    global_data.head(TOP_N).to_csv(
        OUTPUT_DIR / "logistic_global_explanation.csv", index=False
    )

    transformed_features = model_pipeline.named_steps["preprocessor"].transform(
        test_features
    )
    probabilities = model_pipeline.predict_proba(test_features)[:, 1]
    patient_index = probabilities.argmax()
    contributions = transformed_features[patient_index] * coefficients
    local_data = pd.DataFrame(
        {
            "feature": feature_names,
            "transformed_value": transformed_features[patient_index],
            "coefficient": coefficients,
            "contribution": contributions,
        }
    )
    local_data["direction"] = local_data["contribution"].map(
        lambda value: "increases estimated risk" if value > 0 else "decreases estimated risk"
    )
    local_data = local_data.reindex(
        local_data["contribution"].abs().sort_values(ascending=False).index
    )
    local_data.head(TOP_N).to_csv(
        OUTPUT_DIR / "local_logistic_explanation.csv", index=False
    )

    pd.DataFrame(
        {
            "test_row": [patient_index],
            "estimated_readmission_probability": [probabilities[patient_index]],
            "model_intercept": [classifier.intercept_[0]],
        }
    ).to_csv(OUTPUT_DIR / "explained_test_patient.csv", index=False)

    plot_data = global_data.head(10).sort_values("coefficient")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        plot_data["feature"],
        plot_data["coefficient"],
        color=["#c05a4e" if value > 0 else "#2f6f8f" for value in plot_data["coefficient"]],
    )
    axis.set_title("Top Logistic Regression Explanation Features")
    axis.set_xlabel("Coefficient")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "logistic_global_explanation.png", dpi=150)
    plt.close(figure)


def save_forest_explanation(model_pipeline) -> None:
    feature_names = get_feature_names(model_pipeline)
    importances = model_pipeline.named_steps["model"].feature_importances_
    importance_data = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_data.head(TOP_N).to_csv(
        OUTPUT_DIR / "random_forest_global_explanation.csv", index=False
    )

    plot_data = importance_data.head(10).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(plot_data["feature"], plot_data["importance"], color="#4c8c6b")
    axis.set_title("Top Random Forest Importance Features")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "random_forest_global_explanation.png", dpi=150)
    plt.close(figure)


def main() -> None:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}. Run Step 8 first.")

    test_data = pd.read_csv(TEST_PATH)
    test_features = test_data.drop(columns=[TARGET])
    logistic_path = MODEL_DIR / "logistic_regression.joblib"
    forest_path = MODEL_DIR / "random_forest.joblib"
    if not logistic_path.is_file() or not forest_path.is_file():
        raise FileNotFoundError("Trained models not found. Run Step 9 first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logistic_model = joblib.load(logistic_path)
    forest_model = joblib.load(forest_path)
    save_logistic_explanation(logistic_model, test_features)
    save_forest_explanation(forest_model)

    print("=" * 70)
    print("STEP 12: MODEL EXPLAINABILITY")
    print("=" * 70)
    print(f"Top global features saved: {TOP_N} per model")
    print("Local explanation: highest predicted test probability from logistic regression")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nThese explanations describe model behavior, not clinical causation.")
    print("STEP 12 COMPLETE")


if __name__ == "__main__":
    main()
