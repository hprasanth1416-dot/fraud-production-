from pathlib import Path
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Models" / "fraud_model.pkl"
TYPE_MAP = {"online": 0, "pos": 1, "atm": 2}
FEATURES = [
    "amount", "hour", "transaction_type", "location_mismatch",
    "account_age_days", "transactions_last_24h", "average_amount"
]

app = FastAPI(title="FraudShield AI API", version="1.0.0")
model = None
model_load_error = None
if MODEL_PATH.is_file():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        model_load_error = f"Unable to load model from {MODEL_PATH}: {exc}"
        logger.exception("Unable to load fraud model from %s", MODEL_PATH)
else:
    model_load_error = (
        f"Model file not found at {MODEL_PATH}. Run `python train.py` to create it."
    )

class TransactionData(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    transaction_type: str
    location_mismatch: bool
    account_age_days: int = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)
    average_amount: float = Field(ge=0)

@app.get("/")
def home():
    return {"message": "Welcome to FraudShield AI Financial Fraud Detection API"}

@app.get("/health")
def health():
    loaded = model is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_status": "loaded" if loaded else (
            "error" if MODEL_PATH.exists() else "missing"
        ),
        "model_error": model_load_error,
    }

@app.post("/predict")
def predict(transaction: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Fraud model is unavailable.",
        )
    transaction_type = transaction.transaction_type.strip().lower()
    if transaction_type not in TYPE_MAP:
        raise HTTPException(status_code=422, detail="transaction_type must be online, pos, or atm")

    input_data = pd.DataFrame([{
        "amount": transaction.amount,
        "hour": transaction.hour,
        "transaction_type": TYPE_MAP[transaction_type],
        "location_mismatch": int(transaction.location_mismatch),
        "account_age_days": transaction.account_age_days,
        "transactions_last_24h": transaction.transactions_last_24h,
        "average_amount": transaction.average_amount,
    }], columns=FEATURES)

    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        return {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(probability, 4),
            "model_version": "RandomForest-100",
        }
    except Exception as exc:
        logger.exception("Fraud prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for model compatibility details.",
        ) from exc
