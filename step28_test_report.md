# Step 28: System Test Report

Overall: PASS

- [PASS] Valid patient input produces a prediction
- [PASS] Prediction probability is between 0 and 1 (p=0.1820)
- [PASS] Risk category matches selected threshold
- [PASS] Model factors are displayed correctly
- [PASS] Missing required input is handled properly
- [PASS] Invalid categorical input is handled properly
- [PASS] Numerical input validation works
- [PASS] No excluded/leakage feature is accepted
- [PASS] No patient_nbr is required
- [PASS] No encounter_id is required
- [PASS] Existing disclaimer is displayed
- [PASS] Final model loaded by pipeline matches Step 24 selected model (model=gradient_boosting)
- [PASS] Flask application imports successfully
- [PASS] Website loads successfully
- [PASS] Website does not crash on valid input
- [PASS] Website displays disclaimer on valid input
- [PASS] Website displays estimated probability
- [PASS] Missing required input handled on website
- [PASS] Invalid categorical input handled on website
- [PASS] Website does not expose internal errors to the user

STEP 28 STATUS: COMPLETE