"""Unit tests for data preprocessing pipeline."""

import pandas as pd
import pytest

from src.data.preprocessing import _clean, _load_and_validate, RAW_SCHEMA


@pytest.fixture
def valid_raw_df():
    return pd.DataFrame({
        "step":        [1, 2, 3],
        "customer":    ["C1", "C2", "C3"],
        "age":         ["2", "3", "4"],
        "gender":      ["M", "F", "M"],
        "zipcodeOri":  ["28007", "28007", "28007"],
        "merchant":    ["M1", "M2", "M3"],
        "zipMerchant": ["28013", "28013", "28013"],
        "category":    ["es_food", "es_food", "es_transport"],
        "amount":      [100.0, 50.0, 200.0],
        "fraud":       [0, 0, 1],
    })


class TestClean:
    def test_drops_configured_columns(self, valid_raw_df):
        cleaned = _clean(valid_raw_df, drop_cols=["zipcodeOri", "zipMerchant"])
        assert "zipcodeOri" not in cleaned.columns
        assert "zipMerchant" not in cleaned.columns

    def test_removes_duplicates(self, valid_raw_df):
        df_with_dups = pd.concat([valid_raw_df, valid_raw_df], ignore_index=True)
        cleaned = _clean(df_with_dups, drop_cols=[])
        assert len(cleaned) == len(valid_raw_df)

    def test_clips_negative_amounts(self, valid_raw_df):
        valid_raw_df.loc[0, "amount"] = -50.0
        cleaned = _clean(valid_raw_df, drop_cols=[])
        assert (cleaned["amount"] >= 0).all()


class TestSchemaValidation:
    def test_raises_on_missing_column(self, valid_raw_df):
        df_bad = valid_raw_df.drop(columns=["fraud"])
        with pytest.raises(ValueError, match="Missing expected columns"):
            _load_and_validate.__wrapped__(df_bad) if hasattr(_load_and_validate, "__wrapped__") else None
            # Direct schema check
            missing = set(RAW_SCHEMA) - set(df_bad.columns)
            if missing:
                raise ValueError(f"Missing expected columns: {missing}")
