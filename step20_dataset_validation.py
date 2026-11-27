"""Step 20: validate the selected UCI diabetes readmission dataset schema only."""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "uci_diabetes_130_us_hospitals" / "diabetic_data.csv"
REPORT_PATH = PROJECT_ROOT / "analysis" / "step20_dataset_validation_report.md"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "step20_validation_outputs"
TARGET_SOURCE = "readmitted"
DERIVED_TARGET = "readmission_30day"
POSITIVE_LABEL = "<30"

SOURCE_URL = "https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008"
DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"

LEAKAGE_COLUMNS = [
    "discharge_disposition_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

ADMISSION_COLUMNS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Selected dataset not found: {DATA_PATH}. Download/extract the UCI dataset first."
        )
    return pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)


def feature_groups(data: pd.DataFrame) -> dict:
    identifier_columns = [column for column in ["patient_nbr", "encounter_id"] if column in data.columns]
    target_columns = [column for column in [TARGET_SOURCE, DERIVED_TARGET] if column in data.columns]
    date_columns = [
        column
        for column in data.columns
        if "date" in column.lower() or "time" in column.lower()
    ]
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
    return {
        "identifier_columns": identifier_columns,
        "target_columns": target_columns,
        "date_columns": date_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
    }


def save_schema(data: pd.DataFrame, groups: dict) -> None:
    rows = []
    for column in data.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(data[column].dtype),
                "missing_count": int(data[column].isna().sum()),
                "missing_percentage": round(data[column].isna().mean() * 100, 2),
                "unique_count": int(data[column].nunique(dropna=False)),
                "role": (
                    "identifier" if column in groups["identifier_columns"]
                    else "target" if column in groups["target_columns"]
                    else "categorical" if column in groups["categorical_columns"]
                    else "numeric"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "schema_summary.csv", index=False)


def save_plot(target: pd.Series) -> None:
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["No 30-day readmission", "30-day readmission"], counts.values, color=["#2f6f8f", "#c05a4e"])
    axis.set_title("UCI Derived 30-Day Readmission Target")
    axis.set_ylabel("Encounters")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
    plt.close(figure)


def write_report(data: pd.DataFrame, groups: dict, target: pd.Series) -> None:
    missing = data.isna().sum().loc[lambda values: values.gt(0)].sort_values(ascending=False)
    repeated_patients = int(data["patient_nbr"].duplicated().sum())
    lines = [
        "# Step 20: Dataset Validation Report",
        "",
        "## Decision",
        "",
        "**APPROVED WITH TIMING CAVEATS** for the final modeling stage after a documented, leakage-safe schema is created. No model was trained in Step 20.",
        "",
        "## 1. Dataset overview",
        "",
        f"- Dataset: {DATASET_NAME}.",
        f"- Source: {SOURCE_URL}.",
        f"- Rows/encounters: {len(data):,}.",
        f"- Columns: {data.shape[1]}.",
        "- Population: hospitalized patient records with diabetes from 130 US hospitals/integrated delivery networks, 1999-2008, according to UCI documentation.",
        "",
        "## 2. Dataset source",
        "",
        "UCI documents the dataset, its variable table, the introductory paper, the source period, the inclusion criteria, and a CC BY 4.0 license. The project should cite UCI and the associated Strack et al. paper.",
        "",
        "## 3. Target definition",
        "",
        f"- Source target: `{TARGET_SOURCE}`.",
        f"- Observed values: {data[TARGET_SOURCE].value_counts(dropna=False).to_dict()}.",
        f"- Derived final target: `{DERIVED_TARGET} = 1` when `{TARGET_SOURCE} == '{POSITIVE_LABEL}'`; otherwise `0` for documented categories `NO` and `>30`.",
        f"- Derived positives: {int(target.sum()):,} ({target.mean() * 100:.2f}%).",
        "- UCI describes the task as early readmission within 30 days of discharge. The project must preserve the original three-level source target for auditability.",
        "",
        "## 4. Prediction timing",
        "",
        "- A precise timestamp column is **Not documented in the supplied flat file**.",
        "- Admission type/source, demographics, prior utilization counts, and diagnoses are reasonable early candidates subject to the project’s timing definition.",
        "- `discharge_disposition_id` and `time_in_hospital` are discharge/length-of-stay information and should be excluded from an admission-time model.",
        "- Lab/medication counts, procedure counts, HbA1c/glucose result fields, medication change, and diabetes medication status may reflect information collected during the stay; exact availability time is **Not documented**.",
        "",
        "## 5. Patient and encounter structure",
        "",
        f"- Patient ID: `patient_nbr` ({data['patient_nbr'].nunique():,} unique patients).",
        f"- Encounter ID: `encounter_id` ({data['encounter_id'].nunique():,} unique encounters).",
        f"- Repeated patient rows: {repeated_patients:,}.",
        "- Multiple encounters per patient are present, so patient-grouped splitting is required later.",
        "",
        "## 6. Feature groups",
        "",
        "- Demographics: race, gender, age, weight.",
        "- Admission: admission type, admission source, payer, specialty.",
        "- Prior utilization: outpatient, emergency, and inpatient visit counts.",
        "- Diagnoses/comorbidity proxies: diagnosis codes and number of diagnoses.",
        "- Clinical/treatment context: laboratory, procedure, medication, glucose, HbA1c, and diabetes-treatment fields.",
        "- Full column and dtype inventory: `step20_validation_outputs/schema_summary.csv`.",
        "",
        "## 7. Missingness",
        "",
        f"- Columns with missing values after treating `?` as missing: {len(missing)}.",
    ]
    lines.extend(f"- `{column}`: {int(count):,} missing ({count / len(data) * 100:.2f}%)." for column, count in missing.items())
    lines.extend([
        "- UCI documents that missing values exist; the meaning of each missingness pattern must still be considered during preprocessing.",
        "",
        "## 8. Potential leakage",
        "",
        "The following fields require exclusion or explicit timing justification for admission-time prediction:",
    ])
    lines.extend(f"- `{column}`." for column in LEAKAGE_COLUMNS if column in data.columns)
    lines.extend([
        "- The target source column `readmitted` must never be used as a feature.",
        "- `patient_nbr` and `encounter_id` are identifiers, not predictive features.",
        "",
        "## 9. Dataset strengths",
        "",
        "- Explicit documented readmission categories include `<30`.",
        "- Large encounter count and substantially more positive cases than the current 269-case dataset.",
        "- Separate patient and encounter identifiers with repeated encounters.",
        "- Public source with variable documentation, inclusion criteria, paper, and license.",
        "- Prior-utilization variables support a plausible early-risk design.",
        "",
        "## 10. Dataset limitations",
        "",
        "- Exact timestamps and feature availability times are Not documented in the flat file.",
        "- The population is restricted to diabetic encounters and 130 hospitals from 1999-2008.",
        "- The source target includes `>30` and `NO`; converting these to negative labels is a project decision that must be documented.",
        "- Some variables describe care delivered during the hospitalization and may leak post-admission information.",
        "- The dataset is not automatically representative of all hospital patients or current practice.",
        "",
        "## 11. Candidate comparison",
        "",
        "| Dataset | Encounters | Patients | 30-day target | IDs/repeats | Timing/documentation | Suitability |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
        "| Current `sepsis_icu_synthetic` | 5,000 | 5,000 | Binary column, provenance Not documented | Subject ID only; no repeats | Timing and encounter structure Not documented | Reject for final claims |",
        "| UCI Diabetes 130-US Hospitals | 101,766 | 71,518 | `<30`, `>30`, `NO`; explicit early-readmission task | Patient and encounter IDs; repeats present | Strong source/feature documentation; exact timestamps Not documented | Recommended with timing caveats |",
        "| MIMIC-IV | 546,028 hospitalizations; 223,452 hospitalized individuals | 223,452 hospitalized individuals | Not supplied as a ready-made 30-day target; must be derived | Patient/hospitalization/ICU IDs; repeats present | Excellent timestamps and provenance; credentialed access required | Strong future candidate, larger setup burden |",
        "",
        "## 12. Final recommendation",
        "",
        "Select the UCI Diabetes 130-US Hospitals dataset for this project’s replacement schema. It is better documented and structurally defensible than the current synthetic workbook, but the final model must use an admission/early feature subset and preserve the target transformation audit trail.",
        "",
        "## Remaining uncertainties",
        "",
        "- Exact timing for several lab, medication, procedure, and result fields is **Not documented**.",
        "- Exact planned-readmission and incomplete-follow-up handling is **Not documented** in the dataset file.",
        "- The derived binary target is a project transformation from UCI’s three-level target and must be recorded in the final report.",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = feature_groups(data)
    target = (data[TARGET_SOURCE] == POSITIVE_LABEL).astype(int)
    save_schema(data, groups)
    save_plot(target)
    write_report(data, groups, target)

    print("=" * 70)
    print("STEP 20: DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Rows: {len(data):,}; columns: {data.shape[1]}")
    print(f"Patients: {data['patient_nbr'].nunique():,}; encounters: {data['encounter_id'].nunique():,}")
    print(f"Target categories: {data[TARGET_SOURCE].value_counts(dropna=False).to_dict()}")
    print(f"Derived 30-day positives: {int(target.sum()):,} ({target.mean() * 100:.2f}%)")
    print("Models trained: no")
    print("Decision: APPROVED WITH TIMING CAVEATS")
    print(f"Validation report: {REPORT_PATH}")
    print("STEP 20 COMPLETE")


if __name__ == "__main__":
    main()
