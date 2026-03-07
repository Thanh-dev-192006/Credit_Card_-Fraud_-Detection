"""Unit tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import build_features, _add_time_features, _add_amount_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "step":     [1, 5, 10, 1, 5],
        "customer": ["C1", "C1", "C1", "C2", "C2"],
        "merchant": ["M1", "M1", "M2", "M1", "M2"],
        "category": ["es_food", "es_transport", "es_food", "es_food", "es_health"],
        "amount":   [100.0, 50.0, 200.0, 75.0, 30.0],
        "fraud":    [0, 0, 1, 0, 0],
    })


class TestTimeFeatures:
    def test_adds_sin_cos_columns(self, sample_df):
        result = _add_time_features(sample_df)
        assert "step_sin" in result.columns
        assert "step_cos" in result.columns

    def test_cyclical_bounds(self, sample_df):
        result = _add_time_features(sample_df)
        assert result["step_sin"].between(-1, 1).all()
        assert result["step_cos"].between(-1, 1).all()

    def test_time_band_is_integer(self, sample_df):
        result = _add_time_features(sample_df)
        assert result["time_band"].dtype in (np.int8, np.int16, np.int32, np.int64)


class TestAmountFeatures:
    def test_adds_log1p_column(self, sample_df):
        result = _add_amount_features(sample_df)
        assert "amount_log1p" in result.columns
        assert np.allclose(result["amount_log1p"], np.log1p(result["amount"]))

    def test_rolling_mean_monotonically_non_negative(self, sample_df):
        result = _add_amount_features(sample_df)
        assert (result["amount_rolling_mean"] >= 0).all()

    def test_amount_vs_mean_no_inf(self, sample_df):
        result = _add_amount_features(sample_df)
        assert not result["amount_vs_mean"].isin([np.inf, -np.inf]).any()


class TestBuildFeatures:
    def test_target_col_present_in_output(self, sample_df):
        result, _ = build_features(sample_df, fit_encoders=True)
        assert "fraud" in result.columns

    def test_raw_category_col_dropped(self, sample_df):
        result, _ = build_features(sample_df, fit_encoders=True)
        assert "category" not in result.columns

    def test_no_null_features(self, sample_df):
        result, _ = build_features(sample_df, fit_encoders=True)
        assert result.isnull().sum().sum() == 0

    def test_inference_mode_uses_saved_encoders(self, sample_df):
        _, encoders = build_features(sample_df, fit_encoders=True)
        result, _ = build_features(sample_df, fit_encoders=False, encoders=encoders)
        assert "category_encoded" in result.columns
