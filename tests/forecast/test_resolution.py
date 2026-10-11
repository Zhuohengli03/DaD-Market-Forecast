"""Forecasting and scoring a steadier price: 4-hour or daily blocks, plus a typical error that spikes cannot dominate."""
import numpy as np
import pandas as pd
import pytest

from src.forecast.backtest import BacktestResult, aggregate_backtest, summarize
from src.forecast.holdout import run_holdout
from src.forecast.pipeline import ForecastRequest, InvalidRequestError, Season, run_forecast

from .conftest import SEASON_START, make_listings, make_series


def _result():
    actuals = np.array([[10, 12, 14, 16, 20, 20, 20, 20]], dtype=float)
    preds = {"naive": np.full((1, 8), 9.0), "m": np.array([[11, 11, 15, 15, 19, 21, 19, 21]], dtype=float)}
    return BacktestResult(8, (8,), actuals, np.array([9.0]), preds, {}, ())


def test_blocks_average_hours_and_the_baseline_becomes_the_last_blocks_average():
    y = np.array([1, 2, 3, 4, 6, 8, 10, 12, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)   # 8 hours before the origin
    agg = aggregate_backtest(_result(), y, 4)
    assert agg.horizon == 2
    assert np.allclose(agg.actuals, [[13, 20]]) and np.allclose(agg.predictions["m"], [[13, 20]])
    assert np.allclose(agg.predictions["naive"], [[9, 9]])      # mean of hours 4..7 before the origin
    assert np.allclose(agg.last_values, [9])


def test_typical_error_ignores_a_rare_spike():
    actuals = np.array([[10, 10, 10, 10, 10, 10, 10, 30]], dtype=float)   # one spike to 3x
    res = BacktestResult(8, (8,), actuals, np.array([10.0]), {"naive": np.full((1, 8), 10.0)}, {}, ())
    m = summarize(res)[0]
    assert m.mae_mean == pytest.approx(2.5) and m.mdape_mean == pytest.approx(0.0)


def test_forecast_at_4_hours_gives_blocks_and_a_steadier_score():
    load = lambda item, start, end: make_listings(300)
    s = [Season("s2", "Season 2", SEASON_START, None)]
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",), resolution="4h"), load, s)
    assert len(out["forecast"]) == 6 and out["request"]["resolution"] == "4h"
    ts = pd.to_datetime([f["ts"] for f in out["forecast"]])
    assert (ts[1:] - ts[:-1] == pd.Timedelta(hours=4)).all()
    tail = pd.to_datetime([p["ts"] for p in out["series_tail"]])
    assert (tail[1:] - tail[:-1] == pd.Timedelta(hours=4)).all()
    assert "mdape_mean" in out["backtest"]["models"][0]
    with pytest.raises(InvalidRequestError):
        run_forecast(ForecastRequest("Test Ore", "s2", 36, models=("ridge",), resolution="1d"), load, s)


def test_holdout_at_daily_blocks():
    out = run_holdout(make_series(500), SEASON_START, ("naive", "seasonal_naive_24", "ridge"), resolution="1d")
    assert out["resolution"] == "1d" and "mdape" in out["metrics"]
    ts = pd.to_datetime([p["ts"] for p in out["points"]])
    assert (ts[1:] - ts[:-1] >= pd.Timedelta(hours=24)).all()
