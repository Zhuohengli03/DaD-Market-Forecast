from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEASON_START = pd.Timestamp("2026-01-20T00:00:00Z")


def make_listings(hours: int = 240, seed: int = 0, per_hour: int = 5) -> pd.DataFrame:
    """Listings with a daily cycle + slow trend + noise, several per hour."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(hours):
        ts = SEASON_START + pd.Timedelta(hours=h)
        base = 20 + 0.02 * h + 3 * np.sin(2 * np.pi * h / 24)
        for k in range(per_hour):
            price = base + rng.normal(0, 0.5)
            qty = int(rng.integers(1, 7))
            rows.append({
                "item_id": f"{h}-{k}",
                "item": "Test Ore",
                "quantity": qty,
                "price_per_unit": round(price, 2),
                "price": round(price * qty, 2),
                "has_sold": bool(rng.random() < 0.9),
                "created_at": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
            })
    return pd.DataFrame(rows)


def make_series(hours: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(SEASON_START, periods=hours, freq="h", tz="UTC", name="ts")
    h = np.arange(hours)
    price = 20 + 0.02 * h + 3 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.3, hours)
    return pd.DataFrame({
        "median_price": price,
        "min_price": price - 1,
        "max_price": price + 1,
        "p25": price - 0.5,
        "p75": price + 0.5,
        "listings": rng.integers(3, 8, hours),
        "sold": rng.integers(0, 3, hours),
        "volume": rng.integers(5, 30, hours),
    }, index=idx)


@pytest.fixture
def synthetic_listings() -> pd.DataFrame:
    return make_listings()


@pytest.fixture
def synthetic_series() -> pd.DataFrame:
    return make_series()


@pytest.fixture
def season_start() -> pd.Timestamp:
    return SEASON_START
