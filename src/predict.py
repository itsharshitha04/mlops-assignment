"""Load the trained model and make one Iris prediction."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def predict_iris(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
):
    """Predict the Iris species from four flower measurements."""
    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame(
        [
            {
                "sepal length (cm)": sepal_length,
                "sepal width (cm)": sepal_width,
                "petal length (cm)": petal_length,
                "petal width (cm)": petal_width,
            }
        ]
    )

    prediction = model.predict(features)
    return prediction[0]


if __name__ == "__main__":
    result = predict_iris(5.1, 3.5, 1.4, 0.2)
    print(f"Predicted Iris species: {result}")
