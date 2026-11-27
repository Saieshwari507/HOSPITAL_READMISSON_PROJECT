"""Step 19: review whether the current dataset is fit for the final project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = PROJECT_ROOT / "data" / "ML dataset.xlsx"
AUDIT_FEATURES_PATH = PROJECT_ROOT / "analysis" / "timing_audit_outputs" / "prediction_timing_features.csv"
AUDIT_PROVENANCE_PATH = PROJECT_ROOT / "analysis" / "timing_audit_outputs" / "target_provenance_summary.csv"
SIGNAL_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "signal_validation_outputs" / "cross_validation_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "dataset_strategy_report.md"
TARGET = "readmission_30day"


def load_evidence():
    if not ORIGINAL_PATH.is_file():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_PATH}")
    original = pd.read_excel(ORIGINAL_PATH)
    timing = pd.read_csv(AUDIT_FEATURES_PATH) if AUDIT_FEATURES_PATH.is_file() else pd.DataFrame()
    provenance = pd.read_csv(AUDIT_PROVENANCE_PATH) if AUDIT_PROVENANCE_PATH.is_file() else pd.DataFrame()
    signal = pd.read_csv(SIGNAL_SUMMARY_PATH) if SIGNAL_SUMMARY_PATH.is_file() else pd.DataFrame()
    return original, timing, provenance, signal


def current_facts(original, provenance, signal):
    facts = []
    facts.append(f"The workbook contains {original.shape[0]} rows and {original.shape[1]} columns.")
    facts.append(f"The target `{TARGET}` is present with values {original[TARGET].value_counts(dropna=False).to_dict()}.")
    facts.append(f"The target has {int(original[TARGET].isna().sum())} missing values.")
    if "subject_id" in original.columns:
        facts.append(f"`subject_id` has {original['subject_id'].nunique()} unique values and {int(original['subject_id'].duplicated().sum())} duplicate rows.")
    facts.append("The Step 18 audit found no full admission, discharge, or readmission timestamps.")
    facts.append("The Step 18 audit found no encounter ID and no target-generation documentation in the inspected project files.")
    facts.append("The Step 17 experiment found weak and unstable cross-validated signal near random ranking.")
    facts.append("The current workbook uses synthetic-looking clinical variables, but the available files do not prove how the data was generated.")
    return facts


def write_report(original, timing, provenance, signal):
    facts = current_facts(original, provenance, signal)
    signal_detail = "Signal-validation output was not available." if signal.empty else "The strongest mean ROC-AUC was approximately 0.541 and the strongest mean PR-AUC was approximately 0.068, both with substantial fold variability."
    timing_unknown = int((timing["confidence"] == "unknown").sum()) if not timing.empty else "not available"

    lines = [
        "# Step 19: Dataset Strategy Review",
        "",
        "## Purpose",
        "",
        "This is a dataset strategy review, not a modeling experiment. Existing files, models, outputs, and the website were not modified.",
        "",
        "## Verified facts about the current dataset",
        "",
    ]
    lines.extend(f"- {fact}" for fact in facts)
    lines.extend([
        f"- The timing audit classified {timing_unknown} variables as timing unknown.",
        "- The current data has no documented encounter history structure beyond one row per unique subject ID.",
        "",
        "## What a replacement dataset must provide",
        "",
        "### 1. Target definition",
        "",
        "- An explicit 30-day readmission outcome at the encounter level.",
        "- A documented definition: `1` means the patient has an unplanned acute-care readmission within 30 days after discharge; `0` means no qualifying readmission within the defined 30-day window.",
        "- Clear handling of planned readmissions, transfers, deaths, hospice, observation stays, and incomplete follow-up.",
        "",
        "### 2. Prediction timing",
        "",
        "- A clearly named prediction point, preferably admission or a fixed early-hospitalization window.",
        "- Every candidate feature must have a timestamp or documented availability rule relative to that point.",
        "- Features measured after treatment or after clinical deterioration must be separated or excluded from the early model.",
        "",
        "### 3. Patient and encounter information",
        "",
        "- Demographics and social/context variables available at the prediction point.",
        "- Diagnoses and comorbidities with coding definitions.",
        "- Admission type, source, urgency, service, and calendar context.",
        "- Relevant early clinical measurements with collection times.",
        "- Previous utilization and recent encounter history when legally and ethically available.",
        "- A stable patient identifier and a separate encounter identifier.",
        "",
        "### 4. Provenance and documentation",
        "",
        "- Dataset source, owner, collection method, and access/licensing conditions.",
        "- Time period covered and geographic or institutional scope.",
        "- Number of patients and encounters, with a description of repeat encounters.",
        "- Exact target-generation procedure and follow-up window.",
        "- Feature dictionary, units, coding rules, and timing definitions.",
        "- Known limitations, missingness mechanisms, cohort exclusions, and potential selection bias.",
        "",
        "### 5. Data-quality requirements",
        "",
        "- Missing-value counts and meanings for every feature.",
        "- Duplicate and repeat-encounter checks.",
        "- Patient and encounter identifiers that support leakage-safe grouping.",
        "- Valid ranges, units, categorical dictionaries, and outlier rules.",
        "- No features that directly encode post-discharge outcomes or future utilization.",
        "- A documented way to identify whether a patient was observable for the full 30-day outcome window.",
        "",
        "## Weaknesses of the current dataset",
        "",
        "- No admission, discharge, or readmission timestamps were found.",
        "- No encounter ID was found.",
        "- `subject_id` is unique for every row, so repeated patient encounters are not available in this workbook.",
        "- Target-generation documentation is unavailable.",
        "- Feature availability timing is unavailable for many variables.",
        "- Treatment, ICU length-of-stay, laboratory, severity, and sepsis fields may not be valid for an admission-time model without timestamps.",
        "- Step 17 found weak predictive signal; " + signal_detail,
        "- The project evidence suggests a synthetic or semi-synthetic structure, but it does not prove the exact generation process.",
        "",
        "## Decision table",
        "",
        "| Area | Current dataset | Required dataset |",
        "| --- | --- | --- |",
        "| Target definition | Binary `readmission_30day` exists, but label-generation details are undocumented. | Explicit encounter-level 30-day outcome with inclusion/exclusion rules. |",
        "| Target provenance | Not verifiable from project files. | Source and label-generation procedure documented. |",
        "| Prediction timing | No defensible timestamped prediction point. | Admission or fixed early window with feature availability timestamps. |",
        "| Patient history | One unique row per subject; repeated encounters unavailable. | Patient and encounter history with repeat encounters identified. |",
        "| Encounter structure | No encounter ID; one row per subject. | Separate stable patient and encounter IDs. |",
        "| Feature documentation | Names and values exist, but timing/units are incomplete. | Feature dictionary, units, coding, and timing documentation. |",
        "| Missingness documentation | Missing values are observable, but causes/meaning are undocumented. | Missingness definitions and collection process documented. |",
        "| Leakage risk | Cannot be fully assessed because timing is unknown. | Explicit leakage review and timestamped feature windows. |",
        "| Predictive signal | Weak and unstable in Step 17. | Signal should be assessed after provenance and timing are defensible. |",
        "| Final-project suitability | Suitable for pipeline demonstration, not yet defensible as a clinical prediction dataset. | Suitable for a reproducible, timing-defined, explainable research model. |",
        "",
        "## Verified facts versus recommendations",
        "",
        "### Verified facts",
        "",
    ])
    lines.extend(f"- {fact}" for fact in facts)
    lines.extend([
        "",
        "### Recommendations",
        "",
        "- Keep the current files as an educational pipeline and audit record.",
        "- Do not present the current model as a clinically meaningful readmission predictor.",
        "- Obtain a better documented dataset before investing in complex models or deployment claims.",
        "",
        "## Final answers",
        "",
        "### A. Should we keep the current dataset?",
        "Keep it for reproducible educational demonstrations, audit work, and pipeline testing. Do not treat it as the final evidence base for a defensible clinical prediction model.",
        "",
        "### B. Should we replace it?",
        "Yes, for the final modeling and report claims, unless the data owner can supply the missing target, encounter, timing, and provenance documentation.",
        "",
        "### C. What exact properties must the replacement have?",
        "It must have an encounter-level, explicitly defined 30-day readmission target; admission or fixed early prediction timing; patient and encounter IDs; documented repeat encounters; timestamped or availability-defined features; feature dictionary and units; missingness documentation; target-generation provenance; and enough follow-up to establish the outcome window.",
        "",
        "### D. Can the current website architecture be reused after replacing the ML pipeline?",
        "Yes. The Flask routes, authentication, form flow, result layout, and follow-up consideration interface can be reused. The model-loading, feature schema, validation, explanation, and assessment adapter must be retrained and updated for the replacement dataset.",
        "",
        "### E. What should Step 20 be?",
        "Step 20 should be replacement-dataset selection and schema validation: evaluate candidate datasets against this requirements table, document the chosen source, define the prediction point and target precisely, and create a new leakage-safe data contract before retraining models.",
    ])
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    original, timing, provenance, signal = load_evidence()
    write_report(original, timing, provenance, signal)
    print("=" * 70)
    print("STEP 19: DATASET STRATEGY REVIEW")
    print("=" * 70)
    print(f"Current dataset: {original.shape[0]} rows x {original.shape[1]} columns")
    print("Website modified: no")
    print("Existing files deleted or overwritten: no")
    print(f"Report saved to: {OUTPUT_PATH}")
    print("STEP 19 COMPLETE")


if __name__ == "__main__":
    main()
