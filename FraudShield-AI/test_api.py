import os
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
transaction = {
    "amount": 5000,
    "hour": 14,
    "transaction_type": "online",
    "location_mismatch": False,
    "account_age_days": 120,
    "transactions_last_24h": 3,
    "average_amount": 2500
}

def main():
    home = requests.get(f"{BASE_URL}/", timeout=10)
    home.raise_for_status()
    if "message" not in home.json():
        raise AssertionError("Root response is missing API information")

    health = requests.get(f"{BASE_URL}/health", timeout=10)
    print("HEALTH STATUS:", health.status_code, health.json())
    health.raise_for_status()

    if not health.json().get("model_loaded"):
        response = requests.post(
            f"{BASE_URL}/predict", json=transaction, timeout=10
        )
        if response.status_code != 503 or not response.json().get("detail"):
            raise AssertionError(
                "Prediction without a model must return HTTP 503 with an error detail"
            )
        print("API TEST PASSED (model unavailable; prediction returned HTTP 503)")
        return

    response = requests.post(
        f"{BASE_URL}/predict", json=transaction, timeout=10
    )
    print("PREDICTION STATUS:", response.status_code)
    print("PREDICTION RESULT:", response.json())
    response.raise_for_status()

    result = response.json()
    required = {"prediction", "prediction_label", "fraud_probability", "model_version"}
    if not required.issubset(result):
        raise AssertionError("Prediction response is missing required fields")
    if not 0 <= result["fraud_probability"] <= 1:
        raise AssertionError("fraud_probability must be between 0 and 1")
    print("API TEST PASSED")

if __name__ == "__main__":
    main()
