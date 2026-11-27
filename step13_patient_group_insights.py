"""Step 13: build historical patient-group insights."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "feature_engineered_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "patient_group_outputs"
TARGET = "readmission_30day"
MIN_GROUP_SIZE = 30


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {DATA_PATH}. Run Step 7 first."
        )
    return pd.read_csv(DATA_PATH)


def add_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiled = df.copy()
    profiled["profile_older"] = profiled["age"] >= 65
    profiled["profile_high_severity"] = profiled["sofa_score"] >= 9
    profiled["profile_sepsis"] = profiled["sepsis_label"] == 1
    profiled["profile_transfer"] = profiled["hospital_admit_source"] == "Transfer"
    profiled["profile_high_comorbidity"] = profiled["comorbidity_count"] >= 3
    profiled["profile_older_high_severity"] = (
        profiled["profile_older"] & profiled["profile_high_severity"]
    )
    profiled["profile_multiple_risk_signals"] = (
        profiled["profile_older"]
        & profiled["profile_high_severity"]
        & profiled["profile_high_comorbidity"]
    )
    profiled["profile_lower_risk_reference"] = (
        (profiled["age"] < 65)
        & (profiled["sofa_score"] < 9)
        & (profiled["comorbidity_count"] < 3)
    )
    return profiled


def summarize_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        column for column in df.columns if column.startswith("profile_")
    ]
    baseline_rate = df[TARGET].mean()
    rows = []
    for column in profile_columns:
        group = df[df[column]]
        if len(group) < MIN_GROUP_SIZE:
            continue
        rows.append(
            {
                "profile": column[len("profile_") :].replace("_", " "),
                "patient_count": len(group),
                "readmissions": int(group[TARGET].sum()),
                "readmission_rate_percentage": group[TARGET].mean() * 100,
                "risk_ratio_vs_overall": group[TARGET].mean() / baseline_rate,
                "median_age": group["age"].median(),
                "median_sofa_score": group["sofa_score"].median(),
                "median_comorbidity_count": group["comorbidity_count"].median(),
                "missing_lab_rate_percentage": group["lab_missing_count"].gt(0).mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "readmission_rate_percentage", ascending=False
    )


def summarize_bands(df: pd.DataFrame) -> pd.DataFrame:
    profiled = df.copy()
    profiled["severity_band"] = pd.cut(
        profiled["sofa_score"],
        bins=[-1, 5, 8, 12, float("inf")],
        labels=["low", "moderate", "high", "very_high"],
    )
    profiled["comorbidity_band"] = pd.cut(
        profiled["comorbidity_count"],
        bins=[-1, 0, 2, float("inf")],
        labels=["none", "one_to_two", "three_or_more"],
    )
    rows = []
    for column in ["severity_band", "comorbidity_band"]:
        grouped = profiled.groupby(column, observed=False)[TARGET].agg(
            patient_count="count", readmissions="sum", readmission_rate="mean"
        )
        for group_name, values in grouped.iterrows():
            if values["patient_count"] < MIN_GROUP_SIZE:
                continue
            rows.append(
                {
                    "band": column,
                    "group": str(group_name),
                    "patient_count": int(values["patient_count"]),
                    "readmissions": int(values["readmissions"]),
                    "readmission_rate_percentage": values["readmission_rate"] * 100,
                }
            )
    return pd.DataFrame(rows)


def save_report(profiles: pd.DataFrame, bands: pd.DataFrame) -> None:
    lines = [
        "# Step 13: Historical Patient-Group Insights",
        "",
        "Profiles overlap; a patient can belong to more than one profile.",
        "Rates are descriptive historical associations, not clinical recommendations.",
        "",
        "## Profile summary",
        "",
        "| Profile | Patients | Readmissions | Rate | Risk ratio | Median age | Median SOFA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in profiles.iterrows():
        lines.append(
            f"| {row['profile']} | {int(row['patient_count'])} | "
            f"{int(row['readmissions'])} | {row['readmission_rate_percentage']:.2f}% | "
            f"{row['risk_ratio_vs_overall']:.2f} | {row['median_age']:.1f} | "
            f"{row['median_sofa_score']:.1f} |"
        )

    lines.extend(["", "## Band summaries", ""])
    for band_name, band_data in bands.groupby("band"):
        lines.append(f"### {band_name}")
        lines.append("")
        for _, row in band_data.iterrows():
            lines.append(
                f"- {row['group']}: {int(row['patient_count'])} patients, "
                f"{row['readmission_rate_percentage']:.2f}% readmission rate"
            )
        lines.append("")

    (OUTPUT_DIR / "patient_group_insights.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiled = add_profiles(data)
    profiles = summarize_profiles(profiled)
    bands = summarize_bands(profiled)
    profiles.to_csv(OUTPUT_DIR / "profile_summary.csv", index=False)
    bands.to_csv(OUTPUT_DIR / "band_summary.csv", index=False)
    save_report(profiles, bands)

    print("=" * 70)
    print("STEP 13: HISTORICAL PATIENT-GROUP INSIGHTS")
    print("=" * 70)
    print(f"Profiles analyzed: {len(profiles)}")
    print(f"Band groups analyzed: {len(bands)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("STEP 13 COMPLETE")


if __name__ == "__main__":
    main()
