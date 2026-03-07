"""
FastAPI real-time fraud scoring endpoint (future extension scaffold).

Run locally:
    uvicorn api.main:app --reload --port 8000

POST /predict   — score a single transaction
GET  /health    — liveness probe
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Fraud Detection API", version="0.1.0")

# Lazy-load the predictor so the model is only read once at startup
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        import os
        from src.models.predictor import FraudPredictor

        model_name = os.getenv("MODEL_NAME", "lightgbm")
        threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
        artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")
        _predictor = FraudPredictor.from_artifact_dir(artifact_dir, model_name, threshold)
    return _predictor


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, le=180, description="Simulation step (1–180)")
    customer: str
    merchant: str
    category: str
    amount: float = Field(..., ge=0)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v


class TransactionResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: int
    threshold_used: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=TransactionResponse)
def predict(tx: TransactionRequest) -> TransactionResponse:
    """Score a single transaction for fraud probability."""
    import pandas as pd
    from src.features.feature_engineering import build_features
    import pickle

    try:
        predictor = _get_predictor()
        raw = pd.DataFrame([tx.model_dump()])

        # Load saved encoders
        with open("data/features/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        features, _ = build_features(raw, fit_encoders=False, encoders=encoders)
        features = features.drop(columns=["fraud"], errors="ignore")

        prob = float(predictor.predict_proba(features).iloc[0])
        pred = int(prob >= predictor.threshold)

        return TransactionResponse(
            fraud_probability=prob,
            fraud_prediction=pred,
            threshold_used=predictor.threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
