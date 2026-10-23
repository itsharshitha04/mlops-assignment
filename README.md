# MLOps Assignment 4

A small Iris + Random Forest MLOps project demonstrating GitHub Actions CI/CD and an Apache Airflow DAG.

## Project structure
- `data/data.csv`: Iris dataset
- `src/preprocess.py`: cleans data and creates train/test CSV files
- `src/train.py`: trains Random Forest and writes model + metrics
- `src/predict.py`: loads saved model and predicts
- `tests/test_model.py`: basic tests
- `.github/workflows/mlops.yml`: GitHub Actions workflow
- `airflow/dags/mlops_dag.py`: Airflow workflow
- `Dockerfile`: container image for the project

## Local setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
pytest -v
```

If PowerShell blocks activation, use CMD instead:
```bat
.venv\Scripts\activate.bat
```

## GitHub Actions triggers
- Pull request to `main`: flake8 + pytest
- Push to `main`: preprocess + train + upload `models/` artifact
- Push a tag such as `v1.0.0`: train, build and push a container to GitHub Container Registry, then run it

For tag deployment, make sure repository Actions permissions allow the workflow to write packages. GitHub-hosted runners provide `GITHUB_TOKEN`.

## Airflow
Airflow is not natively supported on Windows. Use WSL2/Ubuntu or Docker for local Airflow. Copy `airflow/dags/mlops_dag.py` into your Airflow `dags` folder, set `MLOPS_PROJECT_DIR` to the project directory visible to Airflow, install the project's requirements in the Airflow environment, then start Airflow and trigger `mlops_training_pipeline`.

The DAG is intended for local learning. Its "deployment" copies the best model into a `deployed/` folder; it does not deploy to a real cloud service.


CI pipeline test through pull request.