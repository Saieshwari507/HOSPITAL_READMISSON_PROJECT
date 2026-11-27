"""Step 9: train baseline readmission-risk models."""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET = "readmission_30day"
RANDOM_STATE = 42


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}. Run Step 8 first."
        )

    train_data = pd.read_csv(TRAIN_PATH)
    if TARGET not in train_data.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    features = train_data.drop(columns=[TARGET])
    target = train_data[TARGET]
    preprocessor = build_preprocessor(features)
    models = build_models()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 9: TRAIN MULTIPLE ML MODELS")
    print("=" * 70)
    print(f"Training rows: {len(train_data)}")
    print(f"Positive cases: {int(target.sum())}")
    print("Preprocessing: median imputation, scaling, one-hot encoding")

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(features, target)
        output_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, output_path)
        print(f"Trained and saved: {output_path}")

    print("\nTest data was not used during training.")
    print("STEP 9 COMPLETE")


if __name__ == "__main__":
    main()
