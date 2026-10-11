import json

import numpy as np
import pandas as pd
import pytest

from src.forecast.pipeline import (ForecastRequest, InsufficientDataError, InvalidRequestError,
                                   Season, run_forecast)
from tests.forecast.conftest import SEASON_START, make_listings

S1 = Season("s1", "Season 1", pd.Timestamp("2025-10-06T00:00:00Z"), pd.Timestamp("2025-12-01T00:00:00Z"))
S2 = Season("s2", "Season 2", SEASON_START, None)


def _loader(frames: dict[str, pd.DataFrame]):
    def load(item, start, end):
        df = frames[item]
        ts = pd.to_datetime(df["created_at"], utc=True)
        m = ts >= start
        if end is not None:
            m &= ts < end
        return df[m]
    return load


@pytest.fixture
def frames():
    s2 = make_listings(hours=240, seed=0)
    s1 = make_listings(hours=300, seed=1)
    shift = S1.start - SEASON_START
    s1["created_at"] = pd.to_datetime(s1["created_at"]) + shift
    return {"Test Ore": pd.concat([s1, s2], ignore_index=True)}


def test_end_to_end_schema_and_skill(frames):
    req = ForecastRequest("Test Ore", "s2", horizon_hours=48, models=("ridge", "hgb"))
    logs = []
    out = run_forecast(req, _loader(frames), [S1, S2], source="csv", log=logs.append)
    json.dumps(out)  # serialisable
    for key in ("run_id", "created_at", "request", "source", "series_tail", "forecast", "selected",
                "backtest", "beats_baseline", "skill_vs_naive", "feature_importance", "references", "warnings"):
        assert key in out
    assert len(out["forecast"]) == 48 and out["forecast"][0]["h"] == 1
    f0 = out["forecast"][0]
    assert f0["lower"] <= f0["pred"] <= f0["upper"]
    names = {m["name"] for m in out["backtest"]["models"]}
    assert {"naive", "seasonal_naive_24", "ridge", "hgb"} <= names
    assert out["beats_baseline"] is True and out["skill_vs_naive"] > 0
    assert out["references"] and out["references"][0]["season_id"] == "s1"
    assert out["series_tail"][-1]["ts"] < out["forecast"][0]["ts"]
    assert any("回测" in line for line in logs)
    assert out["backtest"]["coverage"] is None or 0 <= out["backtest"]["coverage"] <= 1


def test_feature_importance_present_for_tree_model(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("rf",)), _loader(frames), [S1, S2])
    assert out["feature_importance"] and "name" in out["feature_importance"][0]


def test_short_last_segment_is_rejected(frames):
    df = frames["Test Ore"]
    ts = pd.to_datetime(df["created_at"], utc=True)
    cut_start = SEASON_START + pd.Timedelta(hours=200)
    cut_end = SEASON_START + pd.Timedelta(hours=210)
    frames2 = {"Test Ore": df[~((ts >= cut_start) & (ts < cut_end))]}   # 10h gap, leaves 30h segment
    with pytest.raises(InsufficientDataError) as e:
        run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames2), [S1, S2])
    assert "30" in str(e.value)


@pytest.mark.parametrize("bad", [
    dict(horizon_hours=12), dict(horizon_hours=200), dict(coverage=0.3),
    dict(models=("ghost",)), dict(season_id="nope"),
])
def test_invalid_requests(frames, bad):
    kwargs = dict(item="Test Ore", season_id="s2", horizon_hours=24, models=("ridge",))
    kwargs.update(bad)
    with pytest.raises(InvalidRequestError):
        run_forecast(ForecastRequest(**kwargs), _loader(frames), [S1, S2])


def test_forecast_bounds_never_negative(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, S2])
    assert all(f["lower"] >= 0 and f["upper"] >= f["lower"] for f in out["forecast"])


def test_refit_failure_falls_back_to_naive(frames, monkeypatch):
    import src.forecast.pipeline as pl
    real = pl.make_model

    def flaky(name):
        m = real(name)
        if name == "ridge":
            orig_fit = m.fit
            def fit(series, start):
                if len(series) >= 240:          # only the final refit sees the whole season
                    raise RuntimeError("singular on full data")
                return orig_fit(series, start)
            m.fit = fit
        return m
    monkeypatch.setattr(pl, "make_model", flaky)
    from src.forecast.select import HonestSelection, Selection
    monkeypatch.setattr(pl, "select_honestly", lambda r: HonestSelection(
        Selection("ridge", (("ridge", 1.0),), True, 0.3, r.predictions["ridge"], ""), r, False, 5, len(r.origins), ()))
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, S2])
    assert out["selected"]["name"] == "naive" and out["beats_baseline"] is False
    assert any("singular" in w for w in out["warnings"])


def test_stale_data_is_flagged(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, S2])
    assert out["data_age_hours"] > 24 * 200            # synthetic data is from Jan 2026
    assert any("小时前" in w for w in out["warnings"])


def test_finished_season_is_not_called_stale(frames):
    ended = Season("s2", "Season 2", SEASON_START, pd.Timestamp("2026-03-01T00:00:00Z"))
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, ended])
    assert out["season"]["ended"] is True
    assert not any("小时前" in w for w in out["warnings"])   # a past season has no newer listings to collect


def test_origin_step_is_respected(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",), backtest_origin_step=3),
                       _loader(frames), [S1, S2])
    out6 = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",), backtest_origin_step=6),
                        _loader(frames), [S1, S2])
    assert out["backtest"]["origins"] > out6["backtest"]["origins"]


def test_refit_fallback_reports_zero_skill(frames, monkeypatch):
    import src.forecast.pipeline as pl
    from src.forecast.select import Selection
    real = pl.make_model

    def flaky(name):
        m = real(name)
        if name == "ridge":
            orig = m.fit
            m.fit = lambda s, st: (_ for _ in ()).throw(RuntimeError("x")) if len(s) >= 240 else orig(s, st)
        return m
    monkeypatch.setattr(pl, "make_model", flaky)
    from src.forecast.select import HonestSelection
    monkeypatch.setattr(pl, "select_honestly", lambda r: HonestSelection(Selection("ridge", (("ridge", 1.0),), True, 0.3, r.predictions["ridge"], ""), r, False, 5, len(r.origins), ()))
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, S2])
    assert out["selected"]["name"] == "naive" and out["skill_vs_naive"] == 0.0


def test_progress_callback_reports_stages(frames):
    events = []
    run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames), [S1, S2],
                 progress=lambda stage, done=None, total=None, **d: events.append((stage, done, total)))
    stages = [e[0] for e in events]
    assert stages[0] == "load" and stages[-1] == "done"
    bt = [e for e in events if e[0] == "backtest"]
    assert bt[-1][1] == bt[-1][2] == 3          # naive, seasonal_naive_24, ridge


def _falling_listings(hours=300, seed=1):
    """Flat at 95, then sliding to 20 during the backtest period; noise about 5% of the price."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(hours):
        level = 95 if h < 0.6 * hours else 95 * (20 / 95) ** ((h - 0.6 * hours) / (0.4 * hours - 1))
        for k in range(4):
            p = level * (1 + rng.normal(0, 0.05))
            rows.append({"item_id": f"{h}-{k}", "item": "Test Ore", "quantity": 1, "price_per_unit": p, "price": p,
                         "has_sold": True, "created_at": SEASON_START + pd.Timedelta(hours=h, minutes=int(rng.integers(0, 60)))})
    return pd.DataFrame(rows)


def test_band_is_sized_for_todays_price_not_the_seasons_average():
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader({"Test Ore": _falling_listings()}), [S2])
    last = out["series_tail"][-1]["median_price"]
    day = out["forecast"][-1]
    assert (day["upper"] - day["lower"]) / last < 0.5             # absolute residuals from the 95 era made it 0.63


def test_backtest_origins_are_spaced_by_horizon_and_report_real_sample_size(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 72, models=("ridge",)), _loader(frames), [S1, S2])
    bt = out["backtest"]
    assert bt["origin_step"] == 18 and 1 <= bt["independent_windows"] <= bt["origins"]
    assert out["request"]["backtest_origin_step"] is None          # automatic unless set


def test_the_method_is_judged_on_backtests_it_was_not_picked_on():
    load = lambda item, start, end: make_listings(hours=500)
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), load, [S2])
    bt = out["backtest"]
    assert bt["in_sample"] is False and bt["evaluation_origins"] >= 5 and bt["selection_origins"] >= 5
    assert bt["selection_origins"] + bt["evaluation_origins"] <= bt["origins"]
