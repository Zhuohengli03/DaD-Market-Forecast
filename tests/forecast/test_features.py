import numpy as np
import pandas as pd

from src.forecast.features import PRICE_LAGS, feature_columns, make_features


def test_feature_columns_match_frame(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    assert list(f.columns) == feature_columns()
    assert f.index.equals(synthetic_series.index)


def test_lags_are_exact(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    price = synthetic_series["median_price"]
    for lag in PRICE_LAGS:
        assert np.allclose(f[f"price_lag_{lag}"].iloc[lag:], price.iloc[:-lag])
        assert f[f"price_lag_{lag}"].iloc[:lag].isna().all()
    assert f["hours_since_start"].iloc[0] == 0 and f["hours_since_start"].iloc[25] == 25


def test_no_same_row_leakage(synthetic_series, season_start):
    base = make_features(synthetic_series, season_start)
    for t in (0, 30, 100, len(synthetic_series) - 1):
        mutated = synthetic_series.copy()
        mutated.iloc[t, mutated.columns.get_loc("median_price")] = 9999.0
        mutated.iloc[t, mutated.columns.get_loc("volume")] = 9999
        mutated.iloc[t, mutated.columns.get_loc("listings")] = 9999
        f = make_features(mutated, season_start)
        pd.testing.assert_series_equal(f.iloc[t], base.iloc[t], check_names=False)


def test_rolling_uses_lagged_price_only(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    price = synthetic_series["median_price"]
    expected = price.shift(1).rolling(6, min_periods=1).mean()
    assert np.allclose(f["price_roll_mean_6"].iloc[1:], expected.iloc[1:])


def test_change_24_handles_zero_denominator(season_start):
    idx = pd.date_range(season_start, periods=30, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": 0.0, "volume": 1, "listings": 1}, index=idx)
    f = make_features(s, season_start)
    assert not np.isinf(f["price_change_24"].fillna(0)).any()
