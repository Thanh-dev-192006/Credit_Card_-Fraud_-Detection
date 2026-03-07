"""
Inference module — loads a saved model and returns fraud probability scores.
Designed to be imported by the future FastAPI scoring endpoint.
"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class FraudPredictor:
    """Wraps a trained sklearn-compatible model for inference."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        with open(model_path, "rb") as f:
            self._model: Any = pickle.load(f)
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return fraud probability for each row."""
        probs = self._model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index, name="fraud_probability")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return binary fraud label using the configured threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int).rename("fraud_prediction")

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str, model_name: str, threshold: float = 0.5) -> "FraudPredictor":
        path = Path(artifact_dir) / "models" / f"{model_name}.pkl"
        return cls(str(path), threshold=threshold)
