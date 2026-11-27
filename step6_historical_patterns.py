"""Step 6: identify historical readmission patterns."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "pattern_outputs"
TARGET = "readmission_30day"
MIN_GROUP_SIZE = 30

CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "insurance",
    "hospital_admit_source",
    "sepsis_label",
]
NUMERIC_COLUMNS = [
    "age",
    "bmi",
    "sofa_score",
    "apache_iv",
    "qsofa",
    "sirs_criteria",
    "gcs_total",
    "icu_los_hours",
    "creatinine",
    "wbc",
    "hemoglobin",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. Run Step 4 first."
        )
    return pd.read_csv(DATA_PATH)


def summarize_groups(
    df: pd.DataFrame, grouping_column: str, grouping_label: str
) -> pd.DataFrame:
    grouped = df.groupby(grouping_column, dropna=False)[TARGET].agg(
        patient_count="count", readmissions="sum", readmission_rate="mean"
    )
    grouped = grouped[grouped["patient_count"] >= MIN_GROUP_SIZE].copy()
    grouped["readmission_rate_percentage"] = grouped["readmission_rate"] * 100
    baseline_rate = df[TARGET].mean()
    grouped["difference_from_baseline_percentage_points"] = (
        grouped["readmission_rate"] - baseline_rate
    ) * 100
    grouped["risk_ratio_vs_overall"] = grouped["readmission_rate"] / baseline_rate
    grouped.insert(0, "pattern_variable", grouping_label)
    grouped.insert(1, "group", grouped.index.astype(str))
    return grouped.reset_index(drop=True).drop(columns=["readmission_rate"])


def categorical_patterns(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            summaries.append(summarize_groups(df, column, column))
    return pd.concat(summaries, ignore_index=True)


def numeric_patterns(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column]
        try:
            bins = pd.qcut(values, q=4, duplicates="drop")
        except ValueError:
            continue
        grouped_data = df.assign(pattern_group=bins).dropna(subset=["pattern_group"])
        summary = summarize_groups(grouped_data, "pattern_group", column)
        summary["group"] = summary["group"].str.replace("(", "[", regex=False)
        summary["group"] = summary["group"].str.replace(")", "]", regex=False)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def save_ranked_patterns(patterns: pd.DataFrame, filename: str) -> None:
    output = patterns.sort_values(
        ["difference_from_baseline_percentage_points", "patient_count"],
        key=lambda values: values.abs() if values.name == "difference_from_baseline_percentage_points" else values,
        ascending=[False, False],
    )
    output.to_csv(OUTPUT_DIR / filename, index=False)


def save_report(df: pd.DataFrame, categorical: pd.DataFrame, numeric: pd.DataFrame) -> None:
    baseline_rate = df[TARGET].mean() * 100
    high_patterns = pd.concat([categorical, numeric], ignore_index=True)
    high_patterns = high_patterns.sort_values(
        "difference_from_baseline_percentage_points", ascending=False
    ).head(10)

    lines = [
        "# Step 6: Historical Readmission Patterns",
        "",
        f"Overall readmission rate: {baseline_rate:.2f}%.",
        f"Only groups with at least {MIN_GROUP_SIZE} patients are included.",
        "",
        "## Highest observed subgroup rates",
        "",
        "| Variable | Group | Patients | Readmission rate | Difference from baseline |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in high_patterns.iterrows():
        lines.append(
            f"| {row['pattern_variable']} | {row['group']} | "
            f"{int(row['patient_count'])} | "
            f"{row['readmission_rate_percentage']:.2f}% | "
            f"{row['difference_from_baseline_percentage_points']:+.2f} percentage points |"
        )
    lines.extend(
        [
            "",
            "These are historical associations, not causal conclusions. "
            "They must be confirmed with a leakage-safe train/test evaluation.",
        ]
    )
    (OUTPUT_DIR / "historical_patterns.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categorical = categorical_patterns(df)
    numeric = numeric_patterns(df)
    save_ranked_patterns(categorical, "categorical_patterns.csv")
    save_ranked_patterns(numeric, "numeric_quartile_patterns.csv")
    save_report(df, categorical, numeric)

    print("=" * 70)
    print("STEP 6: HISTORICAL READMISSION PATTERNS")
    print("=" * 70)
    print(f"Overall readmission rate: {df[TARGET].mean() * 100:.2f}%")
    print(f"Minimum subgroup size: {MIN_GROUP_SIZE}")
    print(f"Categorical patterns analyzed: {len(categorical)}")
    print(f"Numeric quartile patterns analyzed: {len(numeric)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("STEP 6 COMPLETE")


if __name__ == "__main__":
    main()
