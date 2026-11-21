# FraudShield AI — Financial Fraud Detection

This project follows the same simple file layout and workflow as the supplied
Student Performance Prediction project, adapted for synthetic transaction fraud detection.

## Project files

- `train.py`: validates and cleans the 25-row synthetic dataset, then trains a baseline model.
- `mlflow_tracking.py`: trains three Random Forest configurations and logs their runs to MLflow.
- `app.py`: FastAPI prediction service.
- `test_api.py`: checks the running API.
- `Dataset/`: source and cleaned CSV files.
- `Models/`: saved model files.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: container build instructions.

## Run locally (Windows PowerShell)

Create and activate an environment if you do not already have one:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Terminal 1 — start MLflow from the project directory:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000.

Terminal 2 — prepare data and train the baseline:

```powershell
python train.py
```

This creates `Models/fraud_model.pkl`, the model used by the API. MLflow is
optional for running the API. If the model is missing or cannot be loaded, the
API still starts; `/health` reports a degraded status and `/predict` returns
HTTP 503 with the reason.

Terminal 2 — optionally train and log the three MLflow experiments:

```powershell
python mlflow_tracking.py
```

The 100-tree MLflow model is saved as `Models/fraud_model.pkl`, which is loaded by the API.

Terminal 3 — start the API:

```powershell
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs.

Terminal 4 — test the API:

```powershell
python test_api.py
```

## Docker

Build and run after `Models/fraud_model.pkl` has been created:

```powershell
docker build -t fraudshield-ai .
docker run --rm -e PORT=8000 -p 8000:8000 fraudshield-ai
```

The container binds to `0.0.0.0` and uses `PORT` when provided; it defaults to
8000 for local runs.

## Render deployment

The root `render.yaml` defines a Docker web service with `/health` as its health
check. Render's Git integration is configured for automatic deployment only
after CI checks pass. To connect it, push the project to GitHub, then in Render
choose **New > Blueprint**, connect the repository, and apply the `render.yaml`
configuration. Review the service settings and branch in the Render Dashboard.
No deploy hook or production secret is required for the current API. If you add
secret configuration later, enter it in Render's environment settings rather
than committing it.

The GitHub Actions workflow validates the dataset, tests the API in a container,
and builds the image on pushes and pull requests targeting `main`. Render's
automatic deployment is activated by the one-time Blueprint setup; this
workspace has no Git remote and cannot perform that setup itself.

## Sample prediction

```json
{
  "amount": 5000,
  "hour": 14,
  "transaction_type": "online",
  "location_mismatch": false,
  "account_age_days": 120,
  "transactions_last_24h": 3,
  "average_amount": 2500
}
```

## Notes

- The dataset contains exactly 25 synthetic records.
- Results are for learning and demonstration only. Twenty-five records are far too few
  to evaluate or deploy a real financial fraud detection system.
- The API works without a model, but prediction requires `Models/fraud_model.pkl`.
- MLflow Wizard is optional and is not required for traditional scikit-learn tracking.
