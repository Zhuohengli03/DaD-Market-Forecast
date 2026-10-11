"""Fixes from the model review: intervals that scale with price, non-overlapping backtests, no trend extrapolation."""
import numpy as np
import pandas as pd

from src.forecast.backtest import default_origin_step, independent_windows
from src.forecast.intervals import apply_relative_bounds, quantile_bounds, relative_residuals
from src.forecast.models import make_model

from .conftest import SEASON_START, make_series


def test_interval_width_follows_the_price_level():
    # the price fell from 100 to 20; errors were always about 10% of the price
    last = np.array([100.0, 100.0, 100.0, 20.0, 20.0, 20.0] * 4)
    actual = (last * np.tile([1.1, 0.9, 1.0, 1.1, 0.9, 1.0], 4))[:, None]
    pred = last[:, None]
    rel = relative_residuals(actual, pred, last)
    lo, hi = quantile_bounds(rel, 0.9)
    lower, upper = apply_relative_bounds(np.array([20.0]), lo, hi, last_value=20.0)
    # an absolute pool would give about ±10 around 20; relative gives about ±2
    assert 17.5 < lower[0] < 18.5 and 21.5 < upper[0] < 22.5


def test_relative_residuals_skip_non_positive_prices():
    rel = relative_residuals(np.array([[5.0], [5.0]]), np.array([[4.0], [4.0]]), np.array([0.0, 2.0]))
    assert np.isnan(rel[0, 0]) and rel[1, 0] == 0.5


def test_backtest_origins_spread_out_for_long_horizons():
    assert default_origin_step(24) == 6 and default_origin_step(72) == 18 and default_origin_step(168) == 42
    # five 168h windows six hours apart cover barely more than one independent stretch
    assert independent_windows([100, 106, 112, 118, 124], 168) == 1
    assert independent_windows([100, 124, 148], 24) == 3


def test_linear_regression_does_not_extrapolate_the_season_clock():
    s = make_series(300)
    a = make_model("ridge").fit(s, SEASON_START).predict(168)
    b = make_model("ridge").fit(s, SEASON_START - pd.Timedelta(days=60)).predict(168)
    assert np.allclose(a, b)                          # the season start no longer matters to it
    assert "hours_since_start" not in [n for n, _ in make_model("ridge").fit(s, SEASON_START).importance()]
