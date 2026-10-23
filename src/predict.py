"""Load the trained model and make one Iris prediction."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def predict(features):
    model = joblib.load(MODEL_PATH)
    columns = ["sepal length (cm)", "sepal width (cm)",
               "petal length (cm)", "petal width (cm)"]
    sample = pd.DataFrame([features], columns=columns)
    return int(model.predict(sample)[0])


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run python src/preprocess.py and python src/train.py first.")
    print("Predicted Iris class:", predict([5.1, 3.5, 1.4, 0.2]))
