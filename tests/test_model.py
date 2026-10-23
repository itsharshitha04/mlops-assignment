"""Tests for the Iris prediction model."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def test_model_file_exists():
    """Check that the trained model file exists."""
    assert MODEL_PATH.exists()


def test_model_can_predict():
    """Check that the trained model returns a prediction."""
    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame(
        [
            {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        ]
    )

    prediction = model.predict(features)

    assert len(prediction) == 1
