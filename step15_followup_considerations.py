"""Step 15 / 27: conservative follow-up considerations for clinician review."""

from pathlib import Path
import json

import pandas as pd

try:
    from .step26_risk_assessment import assess_patient, example_patient
except ImportError:
    from step26_risk_assessment import assess_patient, example_patient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "followup_outputs"


def generate_considerations(patient_data: dict, assessment: dict) -> list:
    """Return non-prescriptive considerations for qualified clinical review."""
    considerations = []

    if assessment.get("predicted_high_risk") or assessment.get("crosses_screening_threshold"):
        considerations.append(
            "Review the estimated screening result with a qualified clinician; confirm whether it adds value beyond existing judgment."
        )

    age = patient_data.get("age")
    if age is not None:
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            age_value = None
        if age_value is not None and age_value >= 65:
            considerations.append(
                "Consider whether post-discharge contact, home support, and understanding of the care plan should be reviewed."
            )

    prior_inpatient = patient_data.get("number_inpatient")
    if prior_inpatient is not None and int(prior_inpatient) >= 2:
        considerations.append(
            "Prior inpatient utilization is elevated in the submitted snapshot; care coordination and follow-up timing may warrant review."
        )

    prior_emergency = patient_data.get("number_emergency")
    if prior_emergency is not None and int(prior_emergency) >= 1:
        considerations.append(
            "Recent emergency utilization appears in the submitted snapshot; confirm whether discharge planning addresses access and warning signs."
        )

    admission_source = patient_data.get("admission_source_id")
    if admission_source is not None and int(admission_source) in {7, 8}:
        considerations.append(
            "Admission source coding suggests a transfer or referral pattern; confirm handoff information and follow-up ownership if applicable."
        )

    if not considerations:
        considerations.append(
            "Use the standard discharge and follow-up process, with final decisions made by the qualified care team."
        )

    return considerations


def main() -> None:
    example = example_patient()
    assessment = assess_patient(example)
    considerations = generate_considerations(example, assessment)

    result = {
        "assessment": assessment,
        "follow_up_considerations": considerations,
        "disclaimer": assessment["disclaimer"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "example_followup_considerations.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 70)
    print("STEP 15: FOLLOW-UP CONSIDERATIONS")
    print("=" * 70)
    print(f"Considerations generated: {len(considerations)}")
    for number, consideration in enumerate(considerations, start=1):
        print(f"{number}. {consideration}")
    print(f"Output saved to: {output_path}")
    print("STEP 15 COMPLETE")


if __name__ == "__main__":
    main()
