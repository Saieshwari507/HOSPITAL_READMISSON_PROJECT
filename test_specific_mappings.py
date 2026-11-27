"""
Test script to verify specific mapping cases as requested by user
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from step26_risk_assessment import assess_patient

print("=" * 80)
print("TEST A: Urgent -> admission_type_id = 2")
print("=" * 80)

test_a = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 2,  # Urgent
    "admission_source_id": 7,
    "payer_code": "MC",
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_a = assess_patient(test_a)
    print(f"✓ Test A PASSED")
    print(f"  - admission_type_id received: {test_a['admission_type_id']} (expected: 2)")
    print(f"  - Prediction successful: {result_a['estimated_probability']:.4f}")
except Exception as e:
    print(f"✗ Test A FAILED: {e}")

print("\n" + "=" * 80)
print("TEST B: Emergency -> admission_type_id = 1")
print("=" * 80)

test_b = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 1,  # Emergency
    "admission_source_id": 7,
    "payer_code": "MC",
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_b = assess_patient(test_b)
    print(f"✓ Test B PASSED")
    print(f"  - admission_type_id received: {test_b['admission_type_id']} (expected: 1)")
    print(f"  - Prediction successful: {result_b['estimated_probability']:.4f}")
except Exception as e:
    print(f"✗ Test B FAILED: {e}")

print("\n" + "=" * 80)
print("TEST C: Emergency Room -> admission_source_id = 7")
print("=" * 80)

test_c = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 1,
    "admission_source_id": 7,  # Emergency Room
    "payer_code": "MC",
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_c = assess_patient(test_c)
    print(f"✓ Test C PASSED")
    print(f"  - admission_source_id received: {test_c['admission_source_id']} (expected: 7)")
    print(f"  - Prediction successful: {result_c['estimated_probability']:.4f}")
except Exception as e:
    print(f"✗ Test C FAILED: {e}")

print("\n" + "=" * 80)
print("TEST D: Medicare -> payer_code = MD")
print("=" * 80)

test_d = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 1,
    "admission_source_id": 7,
    "payer_code": "MD",  # Medicare
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_d = assess_patient(test_d)
    print(f"✓ Test D PASSED")
    print(f"  - payer_code received: {test_d['payer_code']} (expected: MD)")
    print(f"  - Prediction successful: {result_d['estimated_probability']:.4f}")
except Exception as e:
    print(f"✗ Test D FAILED: {e}")

print("\n" + "=" * 80)
print("TEST E: Medicaid -> payer_code = MC")
print("=" * 80)

test_e = {
    "race": "Caucasian",
    "gender": "Female",
    "age": 75,
    "admission_type_id": 1,
    "admission_source_id": 7,
    "payer_code": "MC",  # Medicaid
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
}

try:
    result_e = assess_patient(test_e)
    print(f"✓ Test E PASSED")
    print(f"  - payer_code received: {test_e['payer_code']} (expected: MC)")
    print(f"  - Prediction successful: {result_e['estimated_probability']:.4f}")
except Exception as e:
    print(f"✗ Test E FAILED: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("All specific mapping tests completed.")
