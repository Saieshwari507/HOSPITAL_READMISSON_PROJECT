"""
UCI Dataset Mappings for Human-Readable Interface

This file contains the official UCI Diabetes 130-US Hospitals dataset mappings
for converting human-readable selections to the original dataset codes expected
by the trained model.

Source: data/uci_diabetes_130_us_hospitals/IDS_mapping.csv
"""

# Admission Type ID Mappings
# Source: IDS_mapping.csv (official UCI dataset documentation)
ADMISSION_TYPE_MAPPING = {
    "Emergency": 1,
    "Urgent": 2,
    "Elective": 3,
    "Newborn": 4,
    "Not Available": 5,
    "NULL": 6,
    "Trauma Center": 7,
    "Not Mapped": 8,
}

# Reverse mapping for display
ADMISSION_TYPE_REVERSE = {v: k for k, v in ADMISSION_TYPE_MAPPING.items()}

# Admission Source ID Mappings
# Source: IDS_mapping.csv (official UCI dataset documentation)
ADMISSION_SOURCE_MAPPING = {
    "Physician Referral": 1,
    "Clinic Referral": 2,
    "HMO Referral": 3,
    "Transfer from a hospital": 4,
    "Transfer from a Skilled Nursing Facility (SNF)": 5,
    "Transfer from another health care facility": 6,
    "Emergency Room": 7,
    "Court/Law Enforcement": 8,
    "Not Available": 9,
    "Transfer from critical access hospital": 10,
    "Normal Delivery": 11,
    "Premature Delivery": 12,
    "Sick Baby": 13,
    "Extramural Birth": 14,
    "Not Available": 15,
    "NULL": 17,
    "Transfer From Another Home Health Agency": 18,
    "Readmission to Same Home Health Agency": 19,
    "Not Mapped": 20,
    "Unknown/Invalid": 21,
    "Transfer from hospital inpt/same fac result in a sep claim": 22,
    "Born inside this hospital": 23,
    "Born outside this hospital": 24,
    "Transfer from Ambulatory Surgery Center": 25,
    "Transfer from Hospice": 26,
}

# Reverse mapping for display
ADMISSION_SOURCE_REVERSE = {v: k for k, v in ADMISSION_SOURCE_MAPPING.items()}

# Payer Code Mappings
# Source: User-provided verified mappings
# Note: These mappings are provided by the user as the authoritative UCI mappings
PAYER_CODE_MAPPING = {
    "Medicaid (MC)": "MC",
    "Medicare (MD)": "MD",
    "Health Maintenance Organization (HM)": "HM",
    "Uninsured (UN)": "UN",
    "Blue Cross/Blue Shield (BC)": "BC",
    "Self-Pay (SP)": "SP",
    "Commercial Payer (CP)": "CP",
    "State Insurance (SI)": "SI",
    "Department of Defense / Military Insurance (DM)": "DM",
    "Charity / Community Program (CM)": "CM",
    "Children's Health Insurance Program (CH)": "CH",
    "Private Insurance - Other (PO)": "PO",
    "Workers' Compensation (WC)": "WC",
    "Other (OT)": "OT",
    "Other Government Programs (OG)": "OG",
    "Managed Care Plan (MP)": "MP",
    "Federal Program (FR)": "FR",
    "Missing/Unknown (?)": "?",
}

# Reverse mapping for display
PAYER_CODE_REVERSE = {v: k for k, v in PAYER_CODE_MAPPING.items()}
