"""Step 17: controlled signal validation using training data only."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "signal_validation_outputs"
TARGET = "readmission_30day"
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3

DEMOGRAPHICS = [
    "age",
    "gender",
    "weight_kg",
    "height_cm",
    "bmi",
    "ethnicity",
    "insurance",
    "age_group",
]
COMORBIDITIES = [
    "diabetes",
    "hypertension",
    "chf",
    "copd",
    "chronic_kidney_disease",
    "liver_disease",
    "immunosuppression",
    "cad",
    "atrial_fibrillation",
    "cancer_active",
    "comorbidity_count",
]
VITALS = [
    "hr_mean",
    "hr_max",
    "hr_min",
    "hr_std",
    "sbp_mean",
    "sbp_max",
    "sbp_min",
    "sbp_std",
    "dbp_mean",
    "dbp_max",
    "dbp_min",
    "dbp_std",
    "map_mean",
    "temp_celsius_mean",
    "temp_celsius_max",
    "temp_celsius_min",
    "temp_celsius_std",
    "spo2_mean",
    "spo2_min",
    "spo2_max",
    "spo2_std",
    "respiratory_rate_mean",
    "respiratory_rate_max",
    "respiratory_rate_min",
    "respiratory_rate_std",
]
LABS = [
    "wbc",
    "lactate_mmol",
    "creatinine",
    "platelet_count",
    "bilirubin_total",
    "glucose",
    "ph_arterial",
    "pao2_fio2_ratio",
    "inr",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "hematocrit",
    "hemoglobin",
]
SEVERITY = ["sofa_score", "apache_iv", "qsofa", "sirs_criteria", "gcs_total"]
ADMISSION_CONTEXT = ["hospital_admit_source", "icu_admit_time_hour", "day_of_week"]
SEPSIS_CONTEXT = ["sepsis_label"]
TREATMENT_VARIABLES = [
    "vasopressors_flag",
    "mechanical_ventilation",
    "fio2_percent",
    "antibiotics_24h",
    "fluids_ml_24h",
    "sedation_score",
    "vasopressor_dose_mcg_kg_min",
    "insulin_infusion_flag",
    "treatment_flag_count",
]
POST_ADMISSION_VARIABLES = ["icu_los_hours"]
MISSINGNESS_FEATURES = ["lab_missing_count", "bmi_missing", "map_missing", "gender_missing"]


FEATURE_GROUPS = {
    "demographics_only": DEMOGRAPHICS,
    "demographics_plus_comorbidities": DEMOGRAPHICS + COMORBIDITIES,
    "admission_early_clinical": (
        DEMOGRAPHICS
        + COMORBIDITIES
        + VITALS
        + LABS
        + SEVERITY
        + ADMISSION_CONTEXT
        + SEPSIS_CONTEXT
        + MISSINGNESS_FEATURES
    ),
    "full_current_feature_set": None,
}


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_logistic_pipeline(features: pd.DataFrame, balanced: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced" if balanced else None,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metric_row(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    model: str,
    feature_group: str,
    repeat: int,
    fold: int,
    prevalence: float,
) -> Dict[str, object]:
    true_negatives = int(((y_true == 0) & (predictions == 0)).sum())
    false_positives = int(((y_true == 0) & (predictions == 1)).sum())
    specificity = true_negatives / (true_negatives + false_positives)
    return {
        "model": model,
        "feature_group": feature_group,
        "repeat": repeat,
        "fold": fold,
        "prevalence": prevalence,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "specificity": specificity,
        "predicted_positive_rate": predictions.mean(),
    }


def available_columns(columns: List[str], data: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in data.columns]


def run_experiment(train_data: pd.DataFrame) -> pd.DataFrame:
    features = train_data.drop(columns=[TARGET])
    target = train_data[TARGET]
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows = []

    for split_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target)
    ):
        repeat = split_number // N_SPLITS + 1
        fold = split_number % N_SPLITS + 1
        fold_features = features.iloc[train_indices]
        fold_target = target.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        validation_target = target.iloc[validation_indices]
        prevalence = fold_target.mean()

        for feature_group, requested_columns in FEATURE_GROUPS.items():
            if requested_columns is None:
                selected_columns = list(features.columns)
            else:
                selected_columns = available_columns(requested_columns, features)
            group_train = fold_features[selected_columns]
            group_validation = validation_features[selected_columns]

            models = {
                "unweighted_logistic_regression": build_logistic_pipeline(
                    group_train, balanced=False
                ),
                "class_balanced_logistic_regression": build_logistic_pipeline(
                    group_train, balanced=True
                ),
            }
            for model_name, model in models.items():
                model.fit(group_train, fold_target)
                probabilities = model.predict_proba(group_validation)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
                rows.append(
                    metric_row(
                        validation_target,
                        probabilities,
                        predictions,
                        model_name,
                        feature_group,
                        repeat,
                        fold,
                        prevalence,
                    )
                )

            majority = DummyClassifier(strategy="most_frequent")
            majority.fit(group_train, fold_target)
            majority_probabilities = majority.predict_proba(group_validation)[:, 1]
            majority_predictions = majority.predict(group_validation)
            rows.append(
                metric_row(
                    validation_target,
                    majority_probabilities,
                    majority_predictions,
                    "majority_class_classifier",
                    feature_group,
                    repeat,
                    fold,
                    prevalence,
                )
            )

            prevalence_probabilities = np.full(len(validation_target), prevalence)
            prevalence_predictions = (prevalence_probabilities >= 0.5).astype(int)
            rows.append(
                metric_row(
                    validation_target,
                    prevalence_probabilities,
                    prevalence_predictions,
                    "prevalence_only_probability",
                    feature_group,
                    repeat,
                    fold,
                    prevalence,
                )
            )

    return pd.DataFrame(rows)


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "recall",
        "precision",
        "f1",
        "specificity",
        "predicted_positive_rate",
    ]
    grouped = fold_results.groupby(["feature_group", "model"])[metric_columns]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def timing_warnings(train_data: pd.DataFrame) -> List[str]:
    available_treatment = available_columns(TREATMENT_VARIABLES, train_data)
    available_post_admission = available_columns(POST_ADMISSION_VARIABLES, train_data)
    lines = [
        "- `icu_los_hours` is likely known only after the encounter has progressed and may be unavailable at admission-time prediction.",
        "- Treatment variables such as vasopressors, ventilation, antibiotics, fluids, sedation, and insulin may occur after clinical deterioration.",
        "- Severity scores, laboratory values, and `sepsis_label` require an explicit timestamp definition before use.",
        "- The admission/early clinical group is a statistical experiment, not proof that these variables are available at the intended prediction point.",
        "- No feature is interpreted as causing readmission; results are associations and model performance measurements only.",
    ]
    lines.append(
        "- Variables present in the current data requiring timing review: "
        + ", ".join(available_post_admission + available_treatment)
        + "."
    )
    return lines


def write_report(
    train_data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    model_results = summary[
        ~summary["model"].isin(
            ["majority_class_classifier", "prevalence_only_probability"]
        )
    ]
    best_roc = model_results.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_pr = model_results.sort_values(
        "average_precision_mean", ascending=False
    ).iloc[0]
    prevalence_ap = float(
        summary.loc[
            summary["model"] == "prevalence_only_probability",
            "average_precision_mean",
        ].iloc[0]
    )
    strongest_group = model_results.groupby("feature_group")[
        "average_precision_mean"
    ].max().sort_values(ascending=False).index[0]
    performance_above_baseline = (
        (model_results["roc_auc_mean"] > 0.5).any()
        and (model_results["average_precision_mean"] > prevalence_ap).any()
    )
    lines = [
        "# Step 17: Signal Validation",
        "",
        "## Design",
        "",
        f"- Training rows: {len(train_data)}.",
        f"- Training positive cases: {int(train_data[TARGET].sum())}.",
        f"- Training prevalence: {train_data[TARGET].mean() * 100:.2f}%.",
        f"- Cross-validation: {N_SPLITS}-fold stratified CV repeated {N_REPEATS} times ({N_SPLITS * N_REPEATS} validation folds).",
        "- The existing held-out test set was not loaded or used by this experiment.",
        "- Baseline model thresholds use 0.5 only for descriptive classification metrics; no threshold was selected.",
        "",
        "## Results",
        "",
        "Metrics are reported as mean +/- standard deviation across validation folds.",
        "",
        "| Feature group | Model | ROC-AUC | PR-AUC | Brier | Recall | Precision | F1 | Specificity | Predicted-positive rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["feature_group", "roc_auc_mean"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['feature_group']} | {row['model']} | "
            f"{row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} | "
            f"{row['average_precision_mean']:.3f} +/- {row['average_precision_std']:.3f} | "
            f"{row['brier_score_mean']:.3f} +/- {row['brier_score_std']:.3f} | "
            f"{row['recall_mean']:.3f} +/- {row['recall_std']:.3f} | "
            f"{row['precision_mean']:.3f} +/- {row['precision_std']:.3f} | "
            f"{row['f1_mean']:.3f} +/- {row['f1_std']:.3f} | "
            f"{row['specificity_mean']:.3f} +/- {row['specificity_std']:.3f} | "
            f"{row['predicted_positive_rate_mean']:.3f} +/- {row['predicted_positive_rate_std']:.3f} |"
        )
    lines.extend(["", "## Timing and leakage review", ""])
    lines.extend(timing_warnings(train_data))
    lines.extend(
        [
            "",
            "## Interpretation framework",
            "",
            "### A. Which baseline performs best?",
            f"Unweighted logistic regression is the strongest probability baseline overall. The best ROC-AUC is {best_roc['roc_auc_mean']:.3f} +/- {best_roc['roc_auc_std']:.3f} for {best_roc['feature_group']}; the best PR-AUC is {best_pr['average_precision_mean']:.3f} +/- {best_pr['average_precision_std']:.3f} for {best_pr['feature_group']}. Class-balanced logistic regression improves recall at threshold 0.5 but has poor Brier scores.",
            "",
            "### B. Is performance consistently above prevalence/random baselines?",
            f"No. Some mean scores exceed the baselines (prevalence PR-AUC {prevalence_ap:.3f}; random ROC-AUC 0.500), but the gains are small and the fold standard deviations are large. This is not consistent evidence of useful generalization ({'some mean metrics are above baseline' if performance_above_baseline else 'mean metrics are not above baseline'}).",
            "",
            "### C. Which feature group contains the most useful signal?",
            f"The strongest group by mean PR-AUC is `{strongest_group}`. Demographics-only has the strongest mean ROC-AUC, while the admission/early-clinical group has the strongest mean PR-AUC. Neither group shows strong or stable predictive performance.",
            "",
            "### D. Is there enough predictive information?",
            "No, not for reliable individual prediction at this stage. The results are close to random ranking and only modestly above the prevalence PR baseline. The dataset may contain weak association, but it does not yet demonstrate enough stable predictive information.",
            "",
            "### E. What should we do next?",
            "First confirm target provenance and prediction-time timestamps, especially for post-admission and treatment variables. Then compare the strongest simple baseline with one conservatively regularized gradient-boosted tabular model using repeated training-only cross-validation. Keep the held-out test set untouched until the final model and threshold are locked.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary[summary["model"].isin(["unweighted_logistic_regression", "class_balanced_logistic_regression"])]
    figure, axis = plt.subplots(figsize=(11, 6))
    for model_name in plot_data["model"].unique():
        model_data = plot_data[plot_data["model"] == model_name]
        axis.errorbar(
            model_data["feature_group"],
            model_data["roc_auc_mean"],
            yerr=model_data["roc_auc_std"],
            marker="o",
            capsize=4,
            label=model_name.replace("_", " ").title(),
        )
    axis.axhline(0.5, color="#555555", linestyle="--", label="Random ranking")
    axis.set_title("Repeated-CV ROC-AUC by Feature Group")
    axis.set_ylabel("Mean ROC-AUC +/- SD")
    axis.set_ylim(0.35, 0.65)
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}. Run Step 8 first.")

    train_data = pd.read_csv(TRAIN_PATH)
    if TARGET not in train_data.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    feature_columns = [column for column in train_data.columns if column != TARGET]
    if TARGET in feature_columns:
        raise ValueError("Target-derived feature detected in the feature columns.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = run_experiment(train_data)
    summary = summarize_results(fold_results)
    fold_results.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cross_validation_summary.csv", index=False)
    save_plot(summary, OUTPUT_DIR / "feature_group_roc_auc.png")
    write_report(train_data, summary, OUTPUT_DIR / "signal_validation_report.md")

    best = summary.sort_values("average_precision_mean", ascending=False).iloc[0]
    print("=" * 70)
    print("STEP 17: SIGNAL VALIDATION")
    print("=" * 70)
    print(f"Training rows: {len(train_data)}")
    print(f"Training prevalence: {train_data[TARGET].mean() * 100:.2f}%")
    print(f"Repeated CV folds: {N_SPLITS * N_REPEATS}")
    print("Held-out test set used: no")
    print(
        "Best mean PR-AUC combination: "
        f"{best['feature_group']} / {best['model']} "
        f"({best['average_precision_mean']:.4f} +/- {best['average_precision_std']:.4f})"
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("STEP 17 COMPLETE")


if __name__ == "__main__":
    main()
