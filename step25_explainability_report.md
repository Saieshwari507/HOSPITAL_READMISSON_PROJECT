# Step 25: Explainability

## A. Global explanation

Method: model feature importances (tree-based).
The table ranks preprocessed features that contribute most overall to model predictions.

                            feature  importance          feature_label
          numeric__number_inpatient    0.863064       number_inpatient
                       numeric__age    0.074667                    age
          numeric__number_emergency    0.030268       number_emergency
    categorical__payer_code_Missing    0.008599     payer_code_Missing
         numeric__number_outpatient    0.004640      number_outpatient
categorical__admission_source_id_17    0.003744 admission_source_id_17
 categorical__admission_source_id_3    0.003168  admission_source_id_3
         categorical__payer_code_OG    0.002294          payer_code_OG
 categorical__admission_source_id_2    0.001714  admission_source_id_2
         categorical__payer_code_BC    0.001176          payer_code_BC
   categorical__admission_type_id_3    0.001117    admission_type_id_3
 categorical__admission_source_id_7    0.001044  admission_source_id_7
 categorical__admission_source_id_8    0.000792  admission_source_id_8
 categorical__admission_source_id_9    0.000695  admission_source_id_9
         categorical__payer_code_CP    0.000611          payer_code_CP

## B. Local explanation

Method: model-attributed directional signal using feature importances and deviation from the local row mean (approximate, not causal).
Example patient (raw approved inputs):

{
  "race": "AfricanAmerican",
  "gender": "Female",
  "age": 75.0,
  "admission_type_id": 1,
  "admission_source_id": 7,
  "payer_code": "MC",
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2
}

Factors model-attributed toward higher estimated risk:

- `number_inpatient` (contribution=0.184017)
- `payer_code_Missing` (contribution=0.008004)
- `number_outpatient` (contribution=0.005685)
- `admission_source_id_7` (contribution=0.000972)
- `admission_type_id_5` (contribution=-0.000000)

Factors model-attributed toward lower estimated risk:

- `age` (contribution=-0.196583)
- `number_emergency` (contribution=-0.008381)
- `admission_source_id_17` (contribution=-0.000259)
- `admission_source_id_3` (contribution=-0.000219)
- `payer_code_OG` (contribution=-0.000159)

## Wording and limitations

- Associations are model attributions, not proof that a feature causes readmission.
- One-hot encoded categories represent patterns in historical data, not deterministic rules.
- Local explanations for tree models use a simplified attribution approach suitable for academic demonstration.

STEP 25 STATUS: COMPLETE