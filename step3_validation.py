# ============================================================
# STEP 3: TARGET & DATA VALIDATION
# Project: Hospital Readmission Risk Prediction
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

# Get the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: TARGET & DATA VALIDATION")
print("=" * 70)

print("\nLoading dataset...")
print(f"Dataset path: {DATA_PATH}")

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    print("\nERROR: Dataset file was not found.")
    print("Make sure your file is located at:")
    print(DATA_PATH)
    raise

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.1 BASIC DATASET INFORMATION")
print("-" * 70)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ------------------------------------------------------------
# 4. CHECK TARGET COLUMN
# ------------------------------------------------------------

TARGET = "readmission_30day"

print("\n" + "-" * 70)
print("3.2 TARGET VARIABLE CHECK")
print("-" * 70)

if TARGET not in df.columns:
    print(f"\nERROR: Target column '{TARGET}' was not found.")
    print("\nAvailable columns:")
    print(list(df.columns))
    raise ValueError(f"Target column '{TARGET}' does not exist.")

print(f"\nTarget column found: {TARGET}")


# ------------------------------------------------------------
# 5. TARGET VALUES
# ------------------------------------------------------------

print("\nUnique target values:")
print(df[TARGET].unique())

print("\nTarget value counts:")
target_counts = df[TARGET].value_counts(dropna=False)

print(target_counts)


# ------------------------------------------------------------
# 6. TARGET PERCENTAGES
# ------------------------------------------------------------

print("\nTarget distribution (%):")

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

for value, percentage in target_percentages.items():
    print(f"{value}: {percentage}%")


# ------------------------------------------------------------
# 7. CHECK TARGET FOR MISSING VALUES
# ------------------------------------------------------------

print("\nMissing target values:")
print(df[TARGET].isna().sum())

if df[TARGET].isna().sum() > 0:
    print("\nWARNING: The target contains missing values.")
else:
    print("No missing target values found.")


# ------------------------------------------------------------
# 8. CHECK TARGET DATA TYPE
# ------------------------------------------------------------

print("\nTarget data type:")
print(df[TARGET].dtype)


# ------------------------------------------------------------
# 9. CHECK SUBJECT ID
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.3 PATIENT / SUBJECT ID CHECK")
print("-" * 70)

if "subject_id" in df.columns:

    total_subjects = df["subject_id"].nunique()
    total_rows = len(df)

    print(f"\nTotal records          : {total_rows}")
    print(f"Unique subject IDs     : {total_subjects}")

    duplicate_subjects = df["subject_id"].duplicated().sum()

    print(f"Repeated subject rows  : {duplicate_subjects}")

    if total_subjects == total_rows:
        print("\nEvery subject_id appears only once.")
        print("This dataset does NOT contain repeated encounters")
        print("for the same patient.")
    else:
        print("\nSome subject IDs appear more than once.")
        print("Patient-level historical encounters may be available.")

    print("\nSubject ID data type:")
    print(df["subject_id"].dtype)

else:
    print("\nWARNING: 'subject_id' column was not found.")


# ------------------------------------------------------------
# 10. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.4 MISSING VALUE CHECK")
print("-" * 70)

missing_counts = df.isnull().sum()

missing_counts = missing_counts[missing_counts > 0]
missing_counts = missing_counts.sort_values(ascending=False)

if len(missing_counts) == 0:

    print("\nNo missing values found in the dataset.")

else:

    print("\nColumns containing missing values:")

    for column, count in missing_counts.items():

        percentage = (count / len(df)) * 100

        print(
            f"{column:35} "
            f"{count:5} missing "
            f"({percentage:6.2f}%)"
        )


# ------------------------------------------------------------
# 11. DATA TYPES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.5 DATA TYPE CHECK")
print("-" * 70)

print("\nData type summary:")

print(df.dtypes.value_counts())

print("\nNumerical columns:")

numerical_columns = df.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns

for column in numerical_columns:
    print(f"- {column}")


print("\nCategorical / text columns:")

categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns

for column in categorical_columns:
    print(f"- {column}")


# ------------------------------------------------------------
# 12. CHECK POTENTIAL ID COLUMNS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.6 POTENTIAL ID / IDENTIFIER COLUMNS")
print("-" * 70)

for column in df.columns:

    unique_count = df[column].nunique(dropna=False)

    if unique_count == len(df):

        print(
            f"{column} -> "
            f"{unique_count} unique values "
            f"(possible identifier)"
        )


# ------------------------------------------------------------
# 13. CHECK SEPSIS LABEL
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.7 SEPSIS LABEL CHECK")
print("-" * 70)

if "sepsis_label" in df.columns:

    print("\nSepsis label values:")
    print(df["sepsis_label"].value_counts(dropna=False))

    print("\nSepsis label percentages:")

    sepsis_percentages = (
        df["sepsis_label"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    print(sepsis_percentages)

else:

    print("\nNo sepsis_label column found.")


# ------------------------------------------------------------
# 14. TARGET RELATIONSHIP WITH SEPSIS LABEL
# ------------------------------------------------------------

if "sepsis_label" in df.columns:

    print("\n" + "-" * 70)
    print("3.8 READMISSION VS SEPSIS LABEL")
    print("-" * 70)

    cross_tab = pd.crosstab(
        df["sepsis_label"],
        df[TARGET]
    )

    print("\nCounts:")
    print(cross_tab)

    print("\nRow percentages:")

    cross_tab_percent = pd.crosstab(
        df["sepsis_label"],
        df[TARGET],
        normalize="index"
    ).mul(100).round(2)

    print(cross_tab_percent)


# ------------------------------------------------------------
# 15. NUMERICAL FEATURE SUMMARY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.9 NUMERICAL FEATURE SUMMARY")
print("-" * 70)

print("\nBasic statistics for numerical columns:")

print(
    df[numerical_columns]
    .describe()
    .T
    .round(2)
)


# ------------------------------------------------------------
# 16. READMISSION RATE BY CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.10 INITIAL READMISSION PATTERN CHECK")
print("-" * 70)

# We only inspect selected categorical variables here.
# We are NOT doing full EDA yet.

categorical_to_check = [
    "gender",
    "ethnicity",
    "insurance",
    "hospital_admit_source",
    "day_of_week"
]

for column in categorical_to_check:

    if column not in df.columns:
        continue

    print(f"\nReadmission rate by: {column}")

    summary = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(
            total_patients="count",
            readmissions="sum",
            readmission_rate="mean"
        )
        .sort_values("readmission_rate", ascending=False)
    )

    summary["readmission_rate"] = (
        summary["readmission_rate"] * 100
    ).round(2)

    print(summary)


# ------------------------------------------------------------
# 17. READMISSION RATE BY SEVERITY SCORE
# ------------------------------------------------------------

severity_columns = [
    "sofa_score",
    "apache_iv",
    "qsofa",
    "sirs_criteria"
]

print("\n" + "-" * 70)
print("3.11 SEVERITY FEATURE CHECK")
print("-" * 70)

for column in severity_columns:

    if column not in df.columns:
        continue

    print(f"\n{column}:")

    print(
        df.groupby(TARGET)[column]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )


# ------------------------------------------------------------
# 18. IMPORTANT: POTENTIAL LEAKAGE REVIEW
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.12 POTENTIAL DATA LEAKAGE REVIEW")
print("-" * 70)

print("""
We need to determine which variables are available at the
moment when the hospital wants to estimate readmission risk.

The following variables require special review:

1. readmission_30day
   -> TARGET ONLY. Never use as an input feature.

2. subject_id
   -> Identifier. Should not be used as an ML feature.

3. sepsis_label
   -> Must be investigated before using it.
      We need to understand whether this label would be
      available at the prediction time.

4. icu_los_hours
   -> Important question:
      Is this known at the prediction point?

5. treatment variables
   -> Some treatment information may only become known
      after the patient's condition has already evolved.

6. laboratory measurements
   -> We need to define when these measurements are taken.

7. severity scores
   -> Need to determine whether they are calculated from
      information available at prediction time.
""")


# ------------------------------------------------------------
# 19. SHOW POTENTIAL FEATURE GROUPS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3.13 FEATURE GROUP OVERVIEW")
print("-" * 70)

feature_groups = {

    "Demographics": [
        "age",
        "gender",
        "weight_kg",
        "height_cm",
        "bmi",
        "ethnicity",
        "insurance"
    ],

    "Vital Signs": [
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
        "respiratory_rate_std"
    ],

    "Laboratory": [
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
        "hemoglobin"
    ],

    "Comorbidities": [
        "diabetes",
        "hypertension",
        "chf",
        "copd",
        "chronic_kidney_disease",
        "liver_disease",
        "immunosuppression",
        "cad",
        "atrial_fibrillation",
        "cancer_active"
    ],

    "Treatment / ICU": [
        "vasopressors_flag",
        "mechanical_ventilation",
        "fio2_percent",
        "antibiotics_24h",
        "fluids_ml_24h",
        "sedation_score",
        "vasopressor_dose_mcg_kg_min",
        "insulin_infusion_flag"
    ],

    "Severity": [
        "sofa_score",
        "apache_iv",
        "qsofa",
        "sirs_criteria",
        "gcs_total"
    ],

    "Hospital / ICU": [
        "icu_los_hours",
        "hospital_admit_source",
        "icu_admit_time_hour",
        "day_of_week"
    ]
}

for group, columns in feature_groups.items():

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    print(f"\n{group} ({len(available)} variables):")

    for column in available:
        print(f"  - {column}")


# ------------------------------------------------------------
# 20. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 SUMMARY")
print("=" * 70)

print(f"""
Dataset:
    Rows                : {df.shape[0]}
    Columns             : {df.shape[1]}

Target:
    {TARGET}

Target counts:
{target_counts.to_string()}

Target percentages:
{target_percentages.to_string()}

Missing-value columns:
    {len(missing_counts)}

Unique patients/subjects:
    {df["subject_id"].nunique() if "subject_id" in df.columns else "N/A"}

Repeated subject rows:
    {df["subject_id"].duplicated().sum() if "subject_id" in df.columns else "N/A"}

Next step:
    Review the output above, especially the potential leakage
    variables and the prediction-time definition.
""")

print("=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)







import pandas as pd
from pathlib import Path

print("=" * 75)
print("STEP 3.14: PREDICTION-TIME VALIDATION")
print("=" * 75)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "ML dataset.xlsx"

print("\nLoading dataset...")
print(f"Dataset path: {data_path}")

df = pd.read_excel(data_path)

print(f"\nDataset loaded successfully!")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ---------------------------------------------------------
# 2. TREATMENT / ICU VARIABLES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.1 TREATMENT / ICU VARIABLES")
print("-" * 75)

treatment_columns = [
    "vasopressors_flag",
    "mechanical_ventilation",
    "fio2_percent",
    "antibiotics_24h",
    "fluids_ml_24h",
    "sedation_score",
    "vasopressor_dose_mcg_kg_min",
    "insulin_infusion_flag",
    "icu_los_hours"
]

for col in treatment_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(f"  Data type : {df[col].dtype}")
        print(f"  Missing   : {df[col].isna().sum()}")
        print(f"  Min       : {df[col].min()}")
        print(f"  Median    : {df[col].median()}")
        print(f"  Mean      : {df[col].mean():.2f}")
        print(f"  Max       : {df[col].max()}")


# ---------------------------------------------------------
# 3. LABORATORY VARIABLES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.2 LABORATORY VARIABLES")
print("-" * 75)

lab_columns = [
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
    "hemoglobin"
]

for col in lab_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(f"  Missing : {df[col].isna().sum()}")
        print(f"  Min     : {df[col].min():.2f}")
        print(f"  Median  : {df[col].median():.2f}")
        print(f"  Mean    : {df[col].mean():.2f}")
        print(f"  Max     : {df[col].max():.2f}")


# ---------------------------------------------------------
# 4. SEVERITY VARIABLES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.3 SEVERITY VARIABLES")
print("-" * 75)

severity_columns = [
    "sofa_score",
    "apache_iv",
    "qsofa",
    "sirs_criteria",
    "gcs_total"
]

for col in severity_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(f"  Missing : {df[col].isna().sum()}")
        print(f"  Min     : {df[col].min():.2f}")
        print(f"  Median  : {df[col].median():.2f}")
        print(f"  Mean    : {df[col].mean():.2f}")
        print(f"  Max     : {df[col].max():.2f}")


# ---------------------------------------------------------
# 5. SEPSIS LABEL
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.4 SEPSIS LABEL")
print("-" * 75)

print("\nValue counts:")
print(df["sepsis_label"].value_counts())

print("\nPercentage:")
print(
    df["sepsis_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ---------------------------------------------------------
# 6. READMISSION RATE BY SEPSIS
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.5 READMISSION RATE BY SEPSIS LABEL")
print("-" * 75)

sepsis_readmission = (
    df.groupby("sepsis_label")["readmission_30day"]
    .agg(
        total_patients="count",
        readmissions="sum",
        readmission_rate="mean"
    )
)

sepsis_readmission["readmission_rate"] *= 100

print(sepsis_readmission.round(2))


# ---------------------------------------------------------
# 7. ICU LENGTH OF STAY VS READMISSION
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.6 ICU LENGTH OF STAY VS READMISSION")
print("-" * 75)

los_summary = (
    df.groupby("readmission_30day")["icu_los_hours"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    )
)

print(los_summary.round(2))


# ---------------------------------------------------------
# 8. TREATMENT VARIABLES VS READMISSION
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.7 TREATMENT VARIABLES VS READMISSION")
print("-" * 75)

binary_treatment_columns = [
    "vasopressors_flag",
    "mechanical_ventilation",
    "antibiotics_24h",
    "insulin_infusion_flag"
]

for col in binary_treatment_columns:

    if col in df.columns:

        print(f"\n{col}")

        result = (
            df.groupby(col)["readmission_30day"]
            .agg(
                total_patients="count",
                readmissions="sum",
                readmission_rate="mean"
            )
        )

        result["readmission_rate"] *= 100

        print(result.round(2))


# ---------------------------------------------------------
# 9. SEVERITY VS READMISSION
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.8 SEVERITY VS READMISSION")
print("-" * 75)

for col in severity_columns:

    if col in df.columns:

        result = (
            df.groupby("readmission_30day")[col]
            .agg(
                mean="mean",
                median="median",
                min="min",
                max="max"
            )
        )

        print(f"\n{col}")
        print(result.round(2))


# ---------------------------------------------------------
# 10. CORRELATION WITH TARGET
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("3.14.9 NUMERICAL CORRELATION WITH READMISSION")
print("-" * 75)

numeric_df = df.select_dtypes(include=["number"])

correlations = (
    numeric_df.corr()["readmission_30day"]
    .drop("readmission_30day")
    .sort_values(key=abs, ascending=False)
)

print("\nTop numerical correlations:")
print(correlations.head(20).round(4))


# ---------------------------------------------------------
# 11. FINAL REVIEW
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 3.14 COMPLETE")
print("=" * 75)

print("""
Important:
This analysis does NOT automatically decide which variables
are safe for the final ML model.

We will use these results to determine:
1. Prediction point
2. Possible leakage
3. Final feature set
4. Variables that should be excluded
5. Variables that require careful justification
""")