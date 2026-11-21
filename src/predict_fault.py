
from pathlib import Path
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MAPPING_FILE = MODELS_DIR / "fault_label_mapping.json"

# All five trained models
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
}

# Must match the training feature order
FEATURES = [
    "TEI", "TCI", "TCO", "TRE", "TRC", "TRC_sub",
    "Tsh_suc", "TR_dis", "TCA", "kW", "TO_sump",
    "PO_net", "PRE", "PRC", "P_lift", "CDW_DT", "CHW_DT",
]


@lru_cache(maxsize=1)
def load_models():
    """Load all five trained models once and reuse them."""

    models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        models[model_name] = joblib.load(model_path)

    return models


@lru_cache(maxsize=1)
def load_label_mapping():
    """Load the fault label mapping once."""

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Label mapping not found: {MAPPING_FILE}")

    with MAPPING_FILE.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def prepare_sensor_readings(sensor_readings):
    """Validate and arrange sensor readings in training order."""

    missing = [
        name for name in FEATURES
        if name not in sensor_readings
    ]

    if missing:
        raise ValueError(f"Missing sensor readings: {missing}")

    row = pd.DataFrame(
        [[sensor_readings[name] for name in FEATURES]],
        columns=FEATURES,
    )

    row = row.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Sensor readings must be finite numeric values.")

    return row


def predict_all_models(sensor_readings):
    """Return one fault prediction from each trained model."""

    row = prepare_sensor_readings(sensor_readings)
    models = load_models()
    mapping = load_label_mapping()

    predictions = {}

    for model_name, model in models.items():
        predicted_id = int(model.predict(row)[0])

        predicted_fault = mapping["id_to_label"][str(predicted_id)]

        predictions[model_name] = predicted_fault

    return predictions


def predict_fault(sensor_readings):
    """Keep the original Logistic Regression prediction function."""

    predictions = predict_all_models(sensor_readings)

    return predictions["Logistic Regression"]


if __name__ == "__main__":
    example_readings = {
        "TEI": 55.0,
        "TCI": 80.0,
        "TCO": 85.0,
        "TRE": 44.2,
        "TRC": 90.0,
        "TRC_sub": 5.0,
        "Tsh_suc": 28.0,
        "TR_dis": 120.0,
        "TCA": 3.5,
        "kW": 130.0,
        "TO_sump": 130.0,
        "PO_net": 10.0,
        "PRE": 44.0,
        "PRC": 105.0,
        "P_lift": 60.0,
        "CDW_DT": 7.0,
        "CHW_DT": 10.0,
    }

    results = predict_all_models(example_readings)

    print("\nPredictions from all five models:")

    for model_name, prediction in results.items():
        print(f"{model_name}: {prediction.upper()}")

