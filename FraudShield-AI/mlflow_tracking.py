from pathlib import Path
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset" / "transactions_cleaned.csv"
MODELS_DIR = ROOT / "Models"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "FraudShield-Experiments"
FEATURES = [
    "amount", "hour", "transaction_type", "location_mismatch",
    "account_age_days", "transactions_last_24h", "average_amount"
]
TYPE_MAP = {"online": 0, "pos": 1, "atm": 2}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python train.py` first to create the cleaned dataset.")

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES].copy()
    X["transaction_type"] = X["transaction_type"].map(TYPE_MAP)
    X["location_mismatch"] = X["location_mismatch"].astype(int)
    y = data["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODELS_DIR.mkdir(exist_ok=True)

    model_files = [
        "fraud_model_1.pkl", "fraud_model.pkl", "fraud_model_2.pkl"
    ]
    for trees, filename in zip([50, 100, 200], model_files):
        model = RandomForestClassifier(n_estimators=trees, random_state=42)
        with mlflow.start_run(run_name=f"random-forest-{trees}") as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
            }
            mlflow.log_params({
                "model_name": "Random Forest",
                "n_estimators": trees,
                "random_state": 42,
                "test_size": 0.2,
                "dataset_rows": len(data),
                "feature_count": len(FEATURES),
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "project": "FraudShield AI",
                "task": "Financial Fraud Detection",
                "dataset": "25 synthetic transactions",
            })

            report_path = ROOT / f"evaluation_{trees}.txt"
            report_path.write_text(
                classification_report(y_test, predictions, zero_division=0)
                + "\\nConfusion matrix:\\n"
                + str(confusion_matrix(y_test, predictions)),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            report_path.unlink(missing_ok=True)

            # Explicit pip requirements avoid slow automatic dependency discovery.
            mlflow.sklearn.log_model(
                sk_model=model,
                name=f"fraud_detection_model_{trees}",
                pip_requirements=[
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "joblib",
                ],
            )
            joblib.dump(model, MODELS_DIR / filename)
            print(f"Trees: {trees}")
            print("Metrics:", metrics)
            print("Run ID:", run.info.run_id)
            print("------------------------------")

    print("All three FraudShield MLflow runs completed.")
    print("Note: metrics from 25 synthetic records are educational only.")

if __name__ == "__main__":
    main()
