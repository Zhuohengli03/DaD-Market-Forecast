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
    assert any("backtest" in line for line in logs)
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
