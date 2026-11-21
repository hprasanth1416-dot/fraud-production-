from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Dataset"
MODELS_DIR = ROOT / "Models"
SOURCE_PATH = DATASET_DIR / "transactions.csv"
CLEAN_PATH = DATASET_DIR / "transactions_cleaned.csv"

FEATURES = [
    "amount", "hour", "transaction_type", "location_mismatch",
    "account_age_days", "transactions_last_24h", "average_amount"
]
TYPE_MAP = {"online": 0, "pos": 1, "atm": 2}

def prepare_data():
    data = pd.read_csv(SOURCE_PATH)
    print("Original dataset shape:", data.shape)
    print(data.head())
    print("\\nMissing values:\\n", data.isnull().sum())
    print("Duplicate rows:", data.duplicated().sum())

    required = FEATURES + ["is_fraud"]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=["is_fraud"]).drop_duplicates().copy()
    for column in ["amount", "hour", "account_age_days",
                   "transactions_last_24h", "average_amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    data["transaction_type"] = data["transaction_type"].astype(str).str.lower()
    if not set(data["transaction_type"]).issubset(TYPE_MAP):
        raise ValueError("transaction_type must be online, pos, or atm")

    data["location_mismatch"] = data["location_mismatch"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    if data["location_mismatch"].isna().any():
        raise ValueError("location_mismatch must contain true/false values")

    data["is_fraud"] = pd.to_numeric(data["is_fraud"], errors="raise").astype(int)
    if not set(data["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1")
    if len(data) != 25:
        raise ValueError(f"Expected exactly 25 records after cleaning; found {len(data)}")
    if data["is_fraud"].nunique() != 2:
        raise ValueError("Dataset must contain both fraud and legitimate classes")

    data.to_csv(CLEAN_PATH, index=False)
    print("\\nCleaned dataset saved:", CLEAN_PATH)
    print("Cleaned shape:", data.shape)
    print("Class distribution:\\n", data["is_fraud"].value_counts())
    return data

def to_features(data):
    X = data[FEATURES].copy()
    X["transaction_type"] = X["transaction_type"].map(TYPE_MAP)
    X["location_mismatch"] = X["location_mismatch"].astype(int)
    return X

def main():
    data = prepare_data()
    X = to_features(data)
    y = data["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Baseline model accuracy:", accuracy_score(y_test, predictions))

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "fraud_model.pkl"
    joblib.dump(model, model_path)
    print("API model saved:", model_path)

if __name__ == "__main__":
    main()
