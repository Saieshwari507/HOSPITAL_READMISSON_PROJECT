"""Flask web application for the hospital readmission research prototype."""

import json
from pathlib import Path
from typing import List

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from analysis.step15_followup_considerations import generate_considerations
from analysis.step26_risk_assessment import APPROVED_FEATURES, assess_patient


PROJECT_ROOT = Path(__file__).resolve().parent
USERS_FILE = PROJECT_ROOT / "users.json"

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-research-prototype-key"


# ---------------------------------------------------------------------------
# Human-readable factor labels
# ---------------------------------------------------------------------------
# Maps internal preprocessed feature name fragments to a plain-English
# sentence that is safe to show to patients.  Categorical one-hot columns
# (e.g. admission_source_id_4, payer_code_SI) are intentionally collapsed
# into a single high-level label so no inactive encoded category is exposed.
_FACTOR_MAP = [
    # numeric utilisation features — checked first (exact or prefix match)
    ("number_inpatient",  "Prior inpatient visit history contributed to the model's estimate."),
    ("number_emergency",  "Prior emergency visit history contributed to the model's estimate."),
    ("number_outpatient", "Prior outpatient visit history contributed to the model's estimate."),
    # age
    ("age",               "Patient age contributed to the model's estimate."),
    # categorical groups — any one-hot column for the group maps to the same
    # high-level phrase so raw category IDs are never surfaced
    ("admission_type_id",   "Admission type information contributed to the model's estimate."),
    ("admission_source_id", "Admission source information contributed to the model's estimate."),
    ("payer_code",          "Payer / insurance information contributed to the model's estimate."),
    ("race",                "Patient demographic information contributed to the model's estimate."),
    ("gender",              "Patient demographic information contributed to the model's estimate."),
]


def summarise_model_factors(raw_factors: List[str]) -> List[str]:
    """Convert raw preprocessed feature names to deduplicated human-readable
    sentences.  Returns at most 4 unique statements.

    The raw names arrive stripped of their pipeline prefix (e.g.
    "number_inpatient", "admission_source_id_4", "payer_code_SI").  We match
    each name against *_FACTOR_MAP* by checking whether the map key is a
    prefix of (or exactly equals) the raw name, so every one-hot variant of
    a categorical group is collapsed to the same sentence.
    """
    seen: set = set()
    result: List[str] = []
    for raw in raw_factors:
        for key, label in _FACTOR_MAP:
            # match exact numeric names and any one-hot variant (key is a
            # leading substring of the raw feature name)
            if raw == key or raw.startswith(key):
                if label not in seen:
                    seen.add(label)
                    result.append(label)
                break
        if len(result) >= 4:
            break
    return result


# ---------------------------------------------------------------------------
# Human-readable labels for the categorical inputs the patient submitted
# ---------------------------------------------------------------------------
_ADMISSION_TYPE_LABELS = {
    "1": "Emergency", "2": "Urgent", "3": "Elective",
    "4": "Newborn", "5": "Not Available", "6": "Not Specified",
    "7": "Trauma Center", "8": "Not Mapped",
}
_ADMISSION_SOURCE_LABELS = {
    "1": "Physician Referral", "2": "Clinic Referral", "3": "HMO Referral",
    "4": "Transfer from a Hospital", "5": "Transfer from Skilled Nursing Facility",
    "6": "Transfer from another Healthcare Facility", "7": "Emergency Room",
    "8": "Court / Law Enforcement", "9": "Not Available",
    "10": "Transfer from Critical Access Hospital",
    "11": "Normal Delivery", "12": "Premature Delivery",
    "13": "Sick Baby", "14": "Extramural Birth", "15": "Not Available",
    "17": "Not Specified", "18": "Transfer from Home Health Agency",
    "19": "Readmission to Home Health Agency", "20": "Not Mapped",
    "21": "Unknown / Invalid",
    "22": "Transfer from Hospital (same facility)",
    "23": "Born Inside this Hospital", "24": "Born Outside this Hospital",
    "25": "Transfer from Ambulatory Surgery Center",
    "26": "Transfer from Hospice",
}
_PAYER_LABELS = {
    "MC": "Medicaid", "MD": "Medicare",
    "HM": "Health Maintenance Organization", "UN": "Uninsured",
    "BC": "Blue Cross / Blue Shield", "SP": "Self-Pay",
    "CP": "Commercial Payer", "SI": "State Insurance",
    "DM": "Department of Defense / Military Insurance",
    "CM": "Charity / Community Program",
    "CH": "Children's Health Insurance Program",
    "PO": "Private Insurance – Other", "WC": "Workers' Compensation",
    "OT": "Other", "OG": "Other Government Programs",
    "MP": "Managed Care Plan", "FR": "Federal Program",
    "?": "Missing / Unknown",
}


def build_patient_context_labels(form_values: dict) -> dict:
    """Return a dict of human-readable strings for the submitted categorical
    values so the template can display them without exposing raw IDs."""
    return {
        "admission_type": _ADMISSION_TYPE_LABELS.get(
            form_values.get("admission_type_id", ""), "Not specified"
        ),
        "admission_source": _ADMISSION_SOURCE_LABELS.get(
            form_values.get("admission_source_id", ""), "Not specified"
        ),
        "payer": _PAYER_LABELS.get(
            form_values.get("payer_code", ""), "Not specified"
        ),
    }


def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


users = load_users()


def parse_patient_form(form):
    patient = {}
    for field in APPROVED_FEATURES:
        value = form.get(field, "")
        if value is None:
            value = ""
        value = str(value).strip()
        if not value:
            continue
        if field in {"admission_type_id", "admission_source_id", "number_outpatient", "number_emergency", "number_inpatient"}:
            try:
                patient[field] = int(value)
            except ValueError as error:
                raise ValueError(f"Enter a valid integer for {field.replace('_', ' ')}.") from error
        elif field == "age":
            if value.startswith("["):
                patient[field] = value
            else:
                try:
                    patient[field] = float(value)
                except ValueError as error:
                    raise ValueError("Enter a valid age in years or a UCI-style age band.") from error
        else:
            patient[field] = value

    missing = [field for field in APPROVED_FEATURES if field not in patient]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    return patient


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = users.get(email)
        if user and check_password_hash(user["password"], password):
            session["user_email"] = email
            return redirect(url_for("index"))
        flash("We could not match that email and password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Enter your name, a valid email, and a password of at least 6 characters.", "error")
        elif email in users:
            flash("An account with that email already exists.", "error")
        else:
            users[email] = {
                "name": name,
                "password": generate_password_hash(password),
            }
            save_users()
            session["user_email"] = email
            return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    if "user_email" not in session:
        return redirect(url_for("login"))

    result = None
    error = None
    form_values = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            patient = parse_patient_form(request.form)
            result = assess_patient(patient)
            result["follow_up_considerations"] = generate_considerations(patient, result)
            factors = result.get("model_attributed_factors", {})
            raw_higher = factors.get("toward_higher_estimated_risk", [])
            result["human_readable_factors"] = summarise_model_factors(raw_higher)
            result["patient_context"] = build_patient_context_labels(form_values)
        except (ValueError, FileNotFoundError) as exception:
            error = str(exception)

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        user=users.get(session["user_email"]),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
