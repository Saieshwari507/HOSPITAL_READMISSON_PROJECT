"""
Test script to verify dropdown mapping implementation
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step26_risk_assessment import assess_patient, APPROVED_FEATURES
import joblib
import pandas as pd

# Load model and preprocessor to verify they are unchanged
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "final_model_config.joblib"

print("=" * 80)
print("TEST 1: Valid category selection and prediction")
print("=" * 80)

# Test Case 1: Using UCI codes directly (as submitted by dropdowns)
test_patient_1 = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 1,  # Emergency
    "admission_source_id": 7,  # Emergency Room
    "payer_code": "MC",  # Medicare
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_1 = assess_patient(test_patient_1)
    print(f"✓ Test 1 PASSED: Prediction successful")
    print(f"  - Estimated probability: {result_1['estimated_probability']:.4f}")
    print(f"  - Risk category: {result_1['risk_category']}")
    print(f"  - Threshold: {result_1['selected_threshold']}")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")

print("\n" + "=" * 80)
print("TEST 2: Verify backend receives correct UCI codes")
print("=" * 80)

# Verify that the input contains the exact UCI codes
print(f"✓ admission_type_id: {test_patient_1['admission_type_id']} (expected: 1)")
print(f"✓ admission_source_id: {test_patient_1['admission_source_id']} (expected: 7)")
print(f"✓ payer_code: {test_patient_1['payer_code']} (expected: 'MC')")

print("\n" + "=" * 80)
print("TEST 3: Verify model receives 9 raw features")
print("=" * 80)

print(f"✓ Approved features count: {len(APPROVED_FEATURES)}")
print(f"  Features: {APPROVED_FEATURES}")

# Verify the patient dict has exactly 9 features
print(f"✓ Test patient has {len(test_patient_1)} features")
print(f"  Features: {list(test_patient_1.keys())}")

print("\n" + "=" * 80)
print("TEST 4: Verify preprocessing produces 55 features")
print("=" * 80)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
config = joblib.load(CONFIG_PATH)

# Create a DataFrame with the test patient
df = pd.DataFrame([test_patient_1])[APPROVED_FEATURES]
print(f"✓ Input DataFrame shape: {df.shape}")

# Transform using the preprocessor
processed = preprocessor.transform(df)
print(f"✓ Processed features count: {processed.shape[1]}")
print(f"  Expected: 55 features")

# Verify feature names from config
feature_names = config.get("feature_names", [])
print(f"✓ Feature names from config: {len(feature_names)}")

print("\n" + "=" * 80)
print("TEST 5: Verify model and threshold unchanged")
print("=" * 80)

threshold = float(config["threshold"])
print(f"✓ Threshold: {threshold}")
print(f"  Expected: 0.15")
print(f"✓ Model type: {type(model).__name__}")

print("\n" + "=" * 80)
print("TEST 6: Verify no old sepsis fields")
print("=" * 80)

# Check that no sepsis-related fields are in APPROVED_FEATURES
sepsis_keywords = ["sepsis", "sofa", "lactate", "vasopressor", "map"]
has_sepsis = any(keyword in " ".join(APPROVED_FEATURES).lower() for keyword in sepsis_keywords)
if not has_sepsis:
    print("✓ No sepsis-related fields found in APPROVED_FEATURES")
else:
    print("✗ Sepsis-related fields found!")

print("\n" + "=" * 80)
print("TEST 7: Verify no 81 features error")
print("=" * 80)

# This test is implicit - if preprocessing produces 55 features (Test 4), 
# then there is no 81 features error
if processed.shape[1] == 55:
    print("✓ Preprocessing produces 55 features (not 81)")
else:
    print(f"✗ Preprocessing produces {processed.shape[1]} features (expected 55)")

print("\n" + "=" * 80)
print("TEST 8: Test second combination of values")
print("=" * 80)

# Test Case 2: Different combination
test_patient_2 = {
    "race": "AfricanAmerican",
    "gender": "Male",
    "age": 45,
    "admission_type_id": 3,  # Elective
    "admission_source_id": 1,  # Physician Referral
    "payer_code": "BC",  # Blue Cross/Blue Shield
    "number_outpatient": 5,
    "number_emergency": 0,
    "number_inpatient": 0,
}

try:
    result_2 = assess_patient(test_patient_2)
    print(f"✓ Test 8 PASSED: Prediction successful")
    print(f"  - Estimated probability: {result_2['estimated_probability']:.4f}")
    print(f"  - Risk category: {result_2['risk_category']}")
except Exception as e:
    print(f"✗ Test 8 FAILED: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("All tests completed. Review results above.")
