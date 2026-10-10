import numpy as np
import pandas as pd
import pytest

from src.forecast.backtest import BacktestResult, plan_origins, run_backtest, summarize
from src.forecast.models import register
from src.forecast.models.base import Forecaster


def test_plan_origins_normal():
    origins, h, warns = plan_origins(n=240, horizon=48, step=6)
    assert origins[0] == 144 and origins[-1] <= 240 - 48 and h == 48 and warns == []


def test_plan_origins_shrinks_horizon():
    origins, h, warns = plan_origins(n=120, horizon=72, step=6, min_origins=5)
    assert len(origins) >= 5 and h < 72 and h >= 24
    assert warns and "缩短" in warns[0]


def test_plan_origins_raises_when_impossible():
    with pytest.raises(ValueError):
        plan_origins(n=60, horizon=24, step=6)


def test_backtest_shapes_and_seasonal_beats_naive(synthetic_series, season_start):
    r = run_backtest(synthetic_series, season_start, ["naive", "seasonal_naive_24"], horizon=24)
    n = len(r.origins)
    assert r.actuals.shape == (n, 24) and r.predictions["naive"].shape == (n, 24)
    assert r.last_values.shape == (n,)
    m = {x.name: x for x in summarize(r)}
    assert m["seasonal_naive_24"].mae_mean < m["naive"].mae_mean
    assert m["naive"].mase_mean == pytest.approx(1.0)
    assert len(m["naive"].per_horizon_mae) == 24
    assert m["naive"].mae_h1 == pytest.approx(m["naive"].per_horizon_mae[0])
    assert np.isnan(m["naive"].direction_acc) and 0 <= m["seasonal_naive_24"].direction_acc <= 1


def test_summary_sorted_by_mae(synthetic_series, season_start):
    r = run_backtest(synthetic_series, season_start, ["naive", "seasonal_naive_24", "ridge"], horizon=24)
    maes = [m.mae_mean for m in summarize(r)]
    assert maes == sorted(maes)


class _Boom(Forecaster):
    name = "boom"

    def fit(self, series, season_start):
        raise RuntimeError("kaboom")


def test_failed_model_is_recorded_not_fatal(synthetic_series, season_start):
    register("boom", _Boom)
    r = run_backtest(synthetic_series, season_start, ["naive", "boom"], horizon=24)
    assert "boom" in r.failures and "kaboom" in r.failures["boom"]
    assert "boom" not in r.predictions and "naive" in r.predictions


def test_mae_h24_is_nan_when_horizon_shorter(synthetic_series, season_start):
    r = run_backtest(synthetic_series.iloc[:120], season_start, ["naive"], horizon=24)
    m = summarize(r)[0]
    assert np.isnan(m.mae_h24) or m.mae_h24 >= 0


def test_filled_hours_are_not_scored(synthetic_series, season_start):
    s = synthetic_series.copy()
    observed = np.ones(len(s), dtype=bool)
    observed[200:204] = False          # pretend these were forward-filled
    r = run_backtest(s, season_start, ["naive"], horizon=24, observed=observed)
    for i, o in enumerate(r.origins):
        for j in range(r.horizon):
            if not observed[o + j]:
                assert np.isnan(r.actuals[i, j])
    assert any("前向填充" in w for w in r.warnings)


def test_direction_accuracy_ignores_ties():
    actual = np.array([[11.0, 9.0, 10.0]])
    r = BacktestResult(3, (0,), actual, np.array([10.0]),
                       {"naive": np.full((1, 3), 10.0), "m": np.array([[12.0, 8.0, 12.0]])}, {}, ())
    m = {x.name: x for x in summarize(r)}
    assert np.isnan(m["naive"].direction_acc)          # naive never predicts a direction
    assert m["m"].direction_acc == 1.0                 # the flat third step is not scored


def test_plan_origins_prefers_more_origins_even_below_minimum():
    origins, h, warns = plan_origins(n=100, horizon=48, step=6, min_origins=5)
    long_origins = list(range(60, 100 - 48 + 1, 6))
    assert len(origins) > len(long_origins) and h < 48
