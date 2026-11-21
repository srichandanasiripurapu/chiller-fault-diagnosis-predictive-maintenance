
from pathlib import Path

import joblib
import numpy as np
import shap

from predict_fault import (
    FEATURES,
    prepare_sensor_readings,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_DIR / "models" / "logistic_regression.joblib"


def explain_prediction(sensor_readings):
    """Explain a Logistic Regression prediction using SHAP."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    # Load the trained Logistic Regression pipeline.
    model = joblib.load(MODEL_FILE)

    # Separate the scaler and classifier.
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]

    # Prepare the sensor readings in the correct feature order.
    row = prepare_sensor_readings(sensor_readings)

    # Apply the same scaling used during model training.
    scaled_row = scaler.transform(row)

    # Standardized zero represents the training feature means.
    background = np.zeros((1, len(FEATURES)))

    # Create the SHAP explainer for the multiclass classifier.
    explainer = shap.LinearExplainer(
        classifier,
        background,
    )

    explanation = explainer(scaled_row)

    return explanation, classifier.classes_
