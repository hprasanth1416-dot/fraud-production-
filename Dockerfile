FROM python:3.12-slim

WORKDIR /app

COPY FraudShield-AI/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY FraudShield-AI/app.py .
COPY FraudShield-AI/Models/ Models/

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
