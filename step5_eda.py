"""Step 5: exploratory data analysis for hospital readmission."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "eda_outputs"
TARGET = "readmission_30day"


CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "insurance",
    "hospital_admit_source",
    "sepsis_label",
]
SEVERITY_COLUMNS = ["sofa_score", "apache_iv", "qsofa", "sirs_criteria", "gcs_total"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. Run Step 4 first."
        )
    return pd.read_csv(DATA_PATH)


def save_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})
    summary.index.name = TARGET
    summary.to_csv(OUTPUT_DIR / "target_distribution.csv")

    figure, axis = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axis, color="#2f6f8f")
    axis.set_title("30-Day Readmission Distribution")
    axis.set_xlabel("Readmission (0 = No, 1 = Yes)")
    axis.set_ylabel("Patients")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
    plt.close(figure)


def save_missingness(df: pd.DataFrame) -> None:
    missing = df.isna().sum().loc[lambda values: values.gt(0)]
    missing = missing.sort_values(ascending=False)
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": (missing / len(df) * 100).round(2),
        }
    )
    summary.to_csv(OUTPUT_DIR / "missing_values.csv")

    if missing.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=summary["missing_percentage"],
        y=summary.index,
        ax=axis,
        color="#c05a4e",
    )
    axis.set_title("Missing Values by Column")
    axis.set_xlabel("Missing values (%)")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "missing_values.png", dpi=150)
    plt.close(figure)


def save_categorical_rates(df: pd.DataFrame) -> None:
    available_columns = [
        column for column in CATEGORICAL_COLUMNS if column in df.columns
    ]
    rows = []
    for column in available_columns:
        grouped = df.groupby(column, dropna=False)[TARGET].agg(
            patient_count="count", readmission_rate="mean"
        )
        for category, values in grouped.iterrows():
            rows.append(
                {
                    "column": column,
                    "category": "Missing" if pd.isna(category) else str(category),
                    "patient_count": values["patient_count"],
                    "readmission_rate_percentage": round(
                        values["readmission_rate"] * 100, 2
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "categorical_readmission_rates.csv", index=False)

    figure, axes = plt.subplots(
        nrows=len(available_columns), ncols=1, figsize=(9, 3 * len(available_columns))
    )
    axes = [axes] if len(available_columns) == 1 else axes
    for axis, column in zip(axes, available_columns):
        plot_data = summary[summary["column"] == column]
        sns.barplot(
            data=plot_data,
            x="readmission_rate_percentage",
            y="category",
            ax=axis,
            color="#4c8c6b",
        )
        axis.set_title(f"Readmission Rate by {column}")
        axis.set_xlabel("Readmission rate (%)")
        axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "categorical_readmission_rates.png", dpi=150)
    plt.close(figure)


def save_severity_comparison(df: pd.DataFrame) -> None:
    available_columns = [column for column in SEVERITY_COLUMNS if column in df.columns]
    summary = (
        df.groupby(TARGET)[available_columns]
        .mean()
        .transpose()
        .rename(columns={0: "not_readmitted_mean", 1: "readmitted_mean"})
    )
    summary.to_csv(OUTPUT_DIR / "severity_comparison.csv")

    plot_data = df.melt(
        id_vars=[TARGET],
        value_vars=available_columns,
        var_name="feature",
        value_name="value",
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_data,
        x="feature",
        y="value",
        hue=TARGET,
        ax=axis,
        palette={0: "#8aa5b5", 1: "#c05a4e"},
    )
    axis.set_title("Severity Features by Readmission Outcome")
    axis.set_xlabel("")
    axis.set_ylabel("Value")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "severity_comparison.png", dpi=150)
    plt.close(figure)


def save_numeric_correlations(df: pd.DataFrame) -> None:
    numeric_data = df.select_dtypes(include="number")
    correlations = (
        numeric_data.corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    correlations.rename("correlation_with_target").to_csv(
        OUTPUT_DIR / "numeric_correlations.csv"
    )

    selected_columns = correlations.head(12).index.tolist() + [TARGET]
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[selected_columns].corr(),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("Top Numeric Correlations with 30-Day Readmission")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "numeric_correlations.png", dpi=150)
    plt.close(figure)


def main() -> None:
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_target_distribution(df)
    save_missingness(df)
    save_categorical_rates(df)
    save_severity_comparison(df)
    save_numeric_correlations(df)

    print("=" * 70)
    print("STEP 5: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"Rows analyzed: {len(df)}")
    print(f"Readmission rate: {df[TARGET].mean() * 100:.2f}%")
    print(f"EDA outputs saved to: {OUTPUT_DIR}")
    print("STEP 5 COMPLETE")


if __name__ == "__main__":
    main()
