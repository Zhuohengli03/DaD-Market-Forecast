"""Season holdout: choose the method on the first 70% of a season, then forecast the rest day by day."""
import numpy as np
import pandas as pd
import pytest

from src.forecast.holdout import HoldoutRequest, run_holdout, run_season_holdout
from src.forecast.pipeline import InsufficientDataError, Season

from .conftest import SEASON_START, make_listings, make_series


def test_holdout_scores_only_the_unseen_last_30_percent():
    series = make_series(400)
    out = run_holdout(series, SEASON_START, ("naive", "seasonal_naive_24", "ridge"), train_frac=0.7, horizon=24)
    split = int(400 * 0.7)
    assert out["split_ts"] == series.index[split].isoformat()
    assert out["train_hours"] == split and out["test_hours"] == 400 - split
    pts = out["points"]
    assert pts[0]["ts"] == series.index[split].isoformat()          # the first scored hour is right after the cut
    assert all(p["ts"] >= out["split_ts"] for p in pts)
    assert len(out["windows"]) == (400 - split) // 24
    m = out["metrics"]
    for key in ("mae", "naive_mae", "skill", "mape", "direction_acc", "coverage"):
        assert key in m
    # the daily cycle is learnable, so the chosen method beats repeating the last hour on unseen data
    assert out["selected"]["name"] != "naive" and m["skill"] > 0 and m["mae"] < m["naive_mae"]
    assert 0 <= m["coverage"] <= 1 and out["nominal_coverage"] == 0.9


def test_the_test_part_never_leaks_into_the_model_choice():
    series = make_series(400)
    a = run_holdout(series, SEASON_START, ("naive", "seasonal_naive_24", "ridge"))
    wrecked = series.copy()
    wrecked.iloc[int(400 * 0.7):, wrecked.columns.get_loc("median_price")] = 1e6   # garbage after the cut
    b = run_holdout(wrecked, SEASON_START, ("naive", "seasonal_naive_24", "ridge"))
    assert a["selected"] == b["selected"] and a["train_skill"] == b["train_skill"]


def test_too_short_a_season_is_refused():
    with pytest.raises(InsufficientDataError):
        run_holdout(make_series(90), SEASON_START, ("naive", "ridge"))


def test_season_holdout_loads_the_season_window():
    listings = make_listings(400)
    season = Season("s1", "Season 1", SEASON_START, SEASON_START + pd.Timedelta(hours=400))
    seen = []

    def load(item, start, end):
        seen.append((item, start, end))
        return listings
    out = run_season_holdout(HoldoutRequest(item="Test Ore", season_id="s1", models=("naive", "seasonal_naive_24")), load, [season])
    assert seen == [("Test Ore", SEASON_START, season.end)]
    assert out["item"] == "Test Ore" and out["season"]["id"] == "s1" and out["run_id"]
    assert np.isfinite(out["metrics"]["mae"])
