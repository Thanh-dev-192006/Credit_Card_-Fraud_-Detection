"""
Feature engineering pipeline for BankSim credit card fraud detection.

Transforms cleaned transaction data into model-ready features covering:
  - Time-based cyclical encodings
  - Customer/merchant behavioural aggregates
  - Log-scaled amount + rolling statistics
  - Target-encoded transaction categories
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full feature engineering pipeline.

    Args:
        df:           Cleaned DataFrame with raw BankSim columns.
        fit_encoders: True during training; False at inference (use saved encoders).
        encoders:     Pre-fitted encoders dict (required when fit_encoders=False).

    Returns:
        (feature_df, encoders) — feature_df has original index preserved.
    """
    encoders = encoders or {}

    df = df.copy()
    df = _add_time_features(df)
    df = _add_amount_features(df)
    df, encoders = _add_frequency_features(df, fit_encoders, encoders)
    df, encoders = _add_category_encoding(df, fit_encoders, encoders)
    df = _drop_raw_cols(df)

    logger.info("Feature matrix shape: %s", df.shape)
    return df, encoders


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical sin/cos encodings of step (1–180, maps to hour-of-week proxy)."""
    max_step = 180.0
    step = df["step"].astype(float)

    return df.assign(
        step_sin=np.sin(2 * np.pi * step / max_step),
        step_cos=np.cos(2 * np.pi * step / max_step),
        # Coarse time bands: morning/afternoon/evening/night
        time_band=pd.cut(
            step % 24,
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            right=False,
        ).cat.codes,  # integer-encoded
    )


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-scaled amount and per-customer rolling statistics."""
    df = df.assign(amount_log1p=np.log1p(df["amount"]))

    # Sort by customer then time so rolling windows are chronologically correct
    df = df.sort_values(["customer", "step"]).reset_index(drop=False)

    rolling = (
        df.groupby("customer")["amount"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )
    rolling_std = (
        df.groupby("customer")["amount"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).std().fillna(0))
    )

    df = df.assign(
        amount_rolling_mean=rolling,
        amount_rolling_std=rolling_std,
        amount_vs_mean=df["amount"] / (rolling + 1e-9),
    )

    return df.sort_values("index").drop(columns="index").reset_index(drop=True)


def _add_frequency_features(
    df: pd.DataFrame,
    fit: bool,
    encoders: dict,
) -> Tuple[pd.DataFrame, dict]:
    """
    Historical transaction counts per customer and per merchant up to each step.
    Uses expanding count (leak-safe: only looks backward).
    """
    df = df.sort_values(["customer", "step"]).reset_index(drop=False)

    df = df.assign(
        customer_tx_count=df.groupby("customer").cumcount(),
        merchant_tx_count=df.groupby("merchant").cumcount(),
        # Ratio: how much of a merchant's volume is from this customer
        customer_merchant_ratio=df.groupby(["customer", "merchant"]).cumcount()
        / (df.groupby("merchant").cumcount() + 1),
    )

    return df.sort_values("index").drop(columns="index").reset_index(drop=True), encoders


def _add_category_encoding(
    df: pd.DataFrame,
    fit: bool,
    encoders: dict,
) -> Tuple[pd.DataFrame, dict]:
    """Target-encode the 'category' column (15 categories in BankSim)."""
    col = "category"

    if fit:
        encoder = TargetEncoder(smooth="auto", random_state=42)
        df = df.assign(
            category_encoded=encoder.fit_transform(
                df[[col]], df["fraud"]
            ).ravel()
        )
        encoders["target_encoder_category"] = encoder
    else:
        encoder = encoders["target_encoder_category"]
        df = df.assign(
            category_encoded=encoder.transform(df[[col]]).ravel()
        )

    return df, encoders


def _drop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns consumed by feature engineering or unusable at inference."""
    cols_to_drop = ["category", "step", "customer", "merchant"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["data"]["processed_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            logger.warning("Split not found, skipping: %s", path)
            continue

        raw_split = pd.read_parquet(path)
        is_train = split == "train"

        if is_train:
            feats, fitted_encoders = build_features(raw_split, fit_encoders=True)
            with open(features_dir / "encoders.pkl", "wb") as f:
                pickle.dump(fitted_encoders, f)
        else:
            with open(features_dir / "encoders.pkl", "rb") as f:
                saved_encoders = pickle.load(f)
            feats, _ = build_features(raw_split, fit_encoders=False, encoders=saved_encoders)

        feats.to_parquet(features_dir / f"{split}_features.parquet", index=False)
        logger.info("Saved %s features: %s rows, %s cols", split, *feats.shape)
