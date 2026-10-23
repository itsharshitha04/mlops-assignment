"""Airflow DAG: download -> preprocess -> train -> evaluate -> deploy/skip."""
from datetime import datetime
import json
import os
import shutil
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_DIR = Path(os.environ.get(
    "MLOPS_PROJECT_DIR", str(Path.home() / "mlops-assignment4")
)).resolve()
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

RAW = PROJECT_DIR / "data" / "data.csv"
PROCESSED = PROJECT_DIR / "data" / "processed"
MODELS = PROJECT_DIR / "models"
DEPLOYED = PROJECT_DIR / "deployed"
BEST = DEPLOYED / "best_metrics.json"


def download_data():
    """Create the Iris CSV locally; avoids depending on a network download."""
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris(as_frame=True)
    df = iris.frame
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    print(f"Prepared Iris dataset: {len(df)} rows at {RAW}")


def preprocess_task():
    from preprocess import preprocess
    preprocess(RAW, PROCESSED)


def train_task():
    from train import train
    train(PROCESSED, MODELS)


def evaluate_task():
    with (MODELS / "metrics.json").open(encoding="utf-8") as f:
        new_acc = json.load(f)["accuracy"]
    old_acc = -1.0
    if BEST.exists():
        with BEST.open(encoding="utf-8") as f:
            old_acc = json.load(f)["accuracy"]
    print(f"new={new_acc:.4f} previous={old_acc:.4f}")
    return "deploy_model" if new_acc > old_acc else "skip_deployment"


def deploy_task():
    DEPLOYED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODELS / "model.joblib", DEPLOYED / "model.joblib")
    shutil.copy2(MODELS / "metrics.json", BEST)
    print(f"Model copied to {DEPLOYED}")


with DAG(
    dag_id="mlops_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "assignment4"],
) as dag:
    download = PythonOperator(task_id="download_data", python_callable=download_data)
    preprocess_op = PythonOperator(task_id="preprocess", python_callable=preprocess_task)
    train_op = PythonOperator(task_id="train_model", python_callable=train_task)
    evaluate = BranchPythonOperator(task_id="evaluate_model", python_callable=evaluate_task)
    deploy = PythonOperator(task_id="deploy_model", python_callable=deploy_task)
    skip = EmptyOperator(task_id="skip_deployment")

    download >> preprocess_op >> train_op >> evaluate >> [deploy, skip]
