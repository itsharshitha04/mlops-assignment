import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import ensure_dataset, preprocess  # noqa: E402
from train import train  # noqa: E402


def test_preprocess_creates_train_and_test_files(tmp_path):
    raw = tmp_path / "data.csv"
    out = tmp_path / "processed"
    ensure_dataset(raw)
    preprocess(raw, out)
    assert (out / "train.csv").exists()
    assert (out / "test.csv").exists()


def test_training_creates_model_and_metrics(tmp_path):
    raw = tmp_path / "data.csv"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    ensure_dataset(raw)
    preprocess(raw, processed)
    metrics = train(processed, models)
    assert 0 <= metrics["accuracy"] <= 1
    assert (models / "model.joblib").exists()
    assert (models / "metrics.json").exists()