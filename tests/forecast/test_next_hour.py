"""Where next hour's individual sales will land, calibrated on the last few days."""
import numpy as np
import pandas as pd
import pytest

from src.forecast.next_hour import next_hour_range
from src.forecast.pipeline import InsufficientDataError

T0 = pd.Timestamp("2026-09-01T00:00:00Z")


def _sales(hours=24 * 10, per_hour=30, level=50.0, spread=0.1, seed=0):
    """Sales scattered around a flat level: price = level * exp(N(0, spread))."""
    rng = np.random.default_rng(seed)
    rows = [{"quantity": 1, "price_per_unit": level * float(np.exp(rng.normal(0, spread))), "has_sold": True,
             "created_at": T0 + pd.Timedelta(hours=h, minutes=int(rng.integers(0, 60)))}
            for h in range(hours) for _ in range(per_hour)]
    return pd.DataFrame(rows)


def test_range_matches_the_spread_of_sales_and_its_hit_rate_is_checked():
    out = next_hour_range(_sales(), now=T0 + pd.Timedelta(hours=240))
    q = out["prices"]
    assert q["p10"] < q["p25"] < q["p50"] < q["p75"] < q["p90"]
    # log-normal spread 0.1: P25..P75 = level x [0.935, 1.07], P10..P90 = level x [0.88, 1.14] (plus the level's own noise)
    assert 45 < q["p25"] < 48.5 and 51.5 < q["p75"] < 55 and 41 < q["p10"] < 45.5 and 55 < q["p90"] < 59.5
    assert out["next_hour"] == (T0 + pd.Timedelta(hours=240)).isoformat()
    assert out["check"]["hours"] >= 100
    assert 0.42 < out["check"]["mid"] < 0.58 and 0.72 < out["check"]["wide"] < 0.88
    assert out["stale_hours"] == pytest.approx(0, abs=1)


def test_old_data_is_flagged_and_too_little_data_is_refused():
    out = next_hour_range(_sales(), now=T0 + pd.Timedelta(hours=240 + 30))
    assert out["stale_hours"] > 24
    with pytest.raises(InsufficientDataError):
        next_hour_range(_sales(hours=6, per_hour=3), now=T0 + pd.Timedelta(hours=6))


def test_unsold_listings_are_not_sales():
    s = _sales(seed=1)
    unsold = s.assign(price_per_unit=s.price_per_unit * 3, has_sold=False)
    a = next_hour_range(s, now=T0 + pd.Timedelta(hours=240))
    b = next_hour_range(pd.concat([s, unsold]), now=T0 + pd.Timedelta(hours=240))
    assert a["prices"] == b["prices"]


def test_the_hit_rate_check_can_be_skipped():
    out = next_hour_range(_sales(), now=T0 + pd.Timedelta(hours=240), check=False)
    assert out["check"] == {"mid": None, "wide": None, "hours": 0}
