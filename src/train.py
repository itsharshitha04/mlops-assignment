"""Train a Random Forest classifier and save model + metrics."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]


def train(processed_dir: str | Path = ROOT / "data" / "processed",
          model_dir: str | Path = ROOT / "models") -> dict:
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    metrics = {
        "accuracy": accuracy,
        "test_rows": int(len(y_test)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Test accuracy: {accuracy:.4f}")
    return metrics


if __name__ == "__main__":
    from preprocess import ensure_dataset, preprocess
    ensure_dataset()
    preprocess()
    train()
