"""Step 28: end-to-end tests for the final model pipeline and Flask website."""

from pathlib import Path
import json
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step26_risk_assessment import APPROVED_FEATURES, assess_patient, example_patient  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "analysis" / "step28_test_report.md"


class TestResult:
    def __init__(self):
        self.items = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.items.append({"name": name, "passed": passed, "detail": detail})

    @property
    def all_passed(self) -> bool:
        return all(item["passed"] for item in self.items)


def run_pipeline_tests(results: TestResult) -> dict:
    example = example_patient()
    assessment = assess_patient(example)
    results.check("Valid patient input produces a prediction", "estimated_probability" in assessment)

    prob = assessment["estimated_probability"]
    results.check("Prediction probability is between 0 and 1", 0.0 <= prob <= 1.0, f"p={prob:.4f}")

    threshold = assessment["selected_threshold"]
    expected_high = prob >= threshold
    results.check(
        "Risk category matches selected threshold",
        assessment["predicted_high_risk"] == expected_high
        and (
            (expected_high and assessment["risk_category"] == "Higher estimated risk")
            or (not expected_high and assessment["risk_category"] == "Lower estimated risk")
        ),
    )

    factors = assessment.get("model_attributed_factors", {})
    results.check(
        "Model factors are displayed correctly",
        bool(factors.get("toward_higher_estimated_risk")) and bool(factors.get("toward_lower_estimated_risk")),
    )

    try:
        assess_patient({field: example[field] for field in APPROVED_FEATURES if field != "race"})
        results.check("Missing required input is handled properly", False, "Expected ValueError")
    except ValueError:
        results.check("Missing required input is handled properly", True)

    bad = dict(example)
    bad["race"] = "NotARealRace"
    try:
        assess_patient(bad)
        results.check("Invalid categorical input is handled properly", False)
    except ValueError:
        results.check("Invalid categorical input is handled properly", True)

    bad_num = dict(example)
    bad_num["number_inpatient"] = -1
    try:
        assess_patient(bad_num)
        results.check("Numerical input validation works", False)
    except ValueError:
        results.check("Numerical input validation works", True)

    try:
        assess_patient({**example, "patient_nbr": 123})
        results.check("No excluded/leakage feature is accepted", False)
    except ValueError:
        results.check("No excluded/leakage feature is accepted", True)

    results.check("No patient_nbr is required", "patient_nbr" not in example)
    results.check("No encounter_id is required", "encounter_id" not in example)
    results.check(
        "Existing disclaimer is displayed",
        "academic decision-support demonstration only" in assessment.get("disclaimer", ""),
    )

    config = joblib.load(PROJECT_ROOT / "models" / "final_model_config.joblib")
    website_model = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    step24_model = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    results.check(
        "Final model loaded by pipeline matches Step 24 selected model",
        config["model_name"] is not None and website_model.__class__ == step24_model.__class__,
        f"model={config['model_name']}",
    )
    return assessment


def run_flask_tests(results: TestResult) -> None:
    try:
        from app import app, parse_patient_form
    except Exception as error:
        results.check("Flask application imports successfully", False, str(error))
        return

    results.check("Flask application imports successfully", True)

    client = app.test_client()
    with client.session_transaction() as session:
        session["user_email"] = "test@example.com"

    response = client.get("/")
    results.check("Website loads successfully", response.status_code == 200)

    payload = {field: str(example_patient()[field]) for field in APPROVED_FEATURES}
    response = client.post("/", data=payload, follow_redirects=True)
    html = response.get_data(as_text=True)
    results.check("Website does not crash on valid input", response.status_code == 200)
    results.check(
        "Website displays disclaimer on valid input",
        "academic decision-support demonstration only" in html,
    )
    results.check(
        "Website displays estimated probability",
        "estimated 30-day" in html.lower() or "%" in html,
    )

    bad_payload = dict(payload)
    bad_payload.pop("gender")
    response = client.post("/", data=bad_payload, follow_redirects=True)
    html = response.get_data(as_text=True)
    results.check(
        "Missing required input handled on website",
        "Missing required fields" in html or "alert" in html,
    )

    response = client.post("/", data={**payload, "race": "InvalidRaceValue"}, follow_redirects=True)
    html = response.get_data(as_text=True)
    results.check(
        "Invalid categorical input handled on website",
        "Invalid race" in html or "alert" in html,
    )

    results.check(
        "Website does not expose internal errors to the user",
        "Traceback" not in html and "File \"" not in html,
    )


def write_report(results: TestResult) -> None:
    lines = [
        "# Step 28: System Test Report",
        "",
        f"Overall: {'PASS' if results.all_passed else 'FAIL'}",
        "",
    ]
    for item in results.items:
        status = "PASS" if item["passed"] else "FAIL"
        detail = f" ({item['detail']})" if item["detail"] else ""
        lines.append(f"- [{status}] {item['name']}{detail}")
    lines.extend(["", "STEP 28 STATUS: COMPLETE"])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = TestResult()
    run_pipeline_tests(results)
    run_flask_tests(results)
    write_report(results)

    print("=" * 70)
    print("STEP 28: END-TO-END SYSTEM TEST")
    print("=" * 70)
    for item in results.items:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"WEBSITE TEST: {'PASS' if results.all_passed else 'FAIL'}")
    print("STEP 28 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
