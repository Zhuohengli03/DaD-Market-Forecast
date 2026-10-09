import numpy as np
import pandas as pd
import pytest

from src.forecast.aggregate import SERIES_COLUMNS, hourly_series, prepare_target, resample_series


def _listing(ts, price, qty=1, sold=True):
    return {"item_id": "x", "item": "Test", "quantity": qty, "price_per_unit": price,
            "price": price * qty, "has_sold": sold, "created_at": ts}


def test_hourly_series_basic_stats():
    rows = [
        _listing("2026-01-01T00:10:00Z", 10.0, 2, True),
        _listing("2026-01-01T00:40:00Z", 20.0, 3, False),
        _listing("2026-01-01T00:50:00Z", 30.0, 1, True),
        _listing("2026-01-01T02:05:00Z", 50.0, 4, True),
    ]
    s = hourly_series(pd.DataFrame(rows))
    assert list(s.columns) == list(SERIES_COLUMNS)
    assert s.index.name == "ts" and str(s.index.tz) == "UTC"
    assert len(s) == 3  # 00:00, 01:00 (empty), 02:00
    h0 = s.iloc[0]
    assert h0["median_price"] == 20.0 and h0["min_price"] == 10.0 and h0["max_price"] == 30.0
    assert h0["listings"] == 3 and h0["sold"] == 2 and h0["volume"] == 6
    h1 = s.iloc[1]
    assert np.isnan(h1["median_price"]) and h1["listings"] == 0 and h1["volume"] == 0


def test_has_sold_accepts_strings_and_bools():
    rows = [_listing("2026-01-01T00:00:00Z", 1.0, sold="True"),
            _listing("2026-01-01T00:01:00Z", 1.0, sold="False"),
            _listing("2026-01-01T00:02:00Z", 1.0, sold=True)]
    s = hourly_series(pd.DataFrame(rows))
    assert s.iloc[0]["sold"] == 2


def test_single_listing_hour_is_its_own_median():
    rows = [_listing("2026-01-01T00:00:00Z", 20.0), _listing("2026-01-01T01:00:00Z", 0.5, qty=2)]
    s = hourly_series(pd.DataFrame(rows))
    assert s.iloc[1]["median_price"] == 0.5


def test_empty_input_gives_empty_frame():
    s = hourly_series(pd.DataFrame(columns=["item_id", "item", "quantity", "price_per_unit",
                                            "price", "has_sold", "created_at"]))
    assert len(s) == 0 and list(s.columns) == list(SERIES_COLUMNS)


def test_prepare_target_fills_short_gaps_and_cuts_long_ones():
    idx = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC", name="ts")
    price = np.arange(40, dtype=float)
    price[5:8] = np.nan          # 3h gap -> filled
    price[20:28] = np.nan        # 8h gap -> cut, keep rows 28..39
    s = pd.DataFrame({"median_price": price, "volume": 1, "listings": 1}, index=idx)
    out, warnings = prepare_target(s, max_gap_hours=6)
    assert out.index[0] == idx[28] and len(out) == 12
    assert not out["median_price"].isna().any()
    assert any("8" in w for w in warnings)


def test_prepare_target_drops_leading_nan():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC", name="ts")
    price = np.arange(10, dtype=float)
    price[:2] = np.nan
    s = pd.DataFrame({"median_price": price, "volume": 1, "listings": 1}, index=idx)
    out, _ = prepare_target(s)
    assert out.index[0] == idx[2] and len(out) == 8


def test_resample_4h_and_1d():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": 1.0, "min_price": 0.5, "max_price": 2.0, "p25": 0.8,
                      "p75": 1.2, "listings": 2, "sold": 1, "volume": 3}, index=idx)
    r4 = resample_series(s, "4h")
    assert len(r4) == 12 and r4.iloc[0]["volume"] == 12 and r4.iloc[0]["min_price"] == 0.5
    r1d = resample_series(s, "1d")
    assert len(r1d) == 2 and r1d.iloc[0]["listings"] == 48
    assert resample_series(s, "1h").equals(s)
    with pytest.raises(ValueError):
        resample_series(s, "2h")
