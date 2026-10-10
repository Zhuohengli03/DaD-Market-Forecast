"""Features for hour t that only read rows <= t-1 (plus calendar/offset of t itself)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
EXOG_LAGS: tuple[int, ...] = (1, 6, 24)
ROLL_WINDOWS: tuple[int, ...] = (6, 24)
MAX_LAG = max(PRICE_LAGS)


def feature_columns() -> list[str]:
    cols = [f"price_lag_{l}" for l in PRICE_LAGS]
    for l in EXOG_LAGS:
        cols += [f"volume_lag_{l}", f"listings_lag_{l}"]
    for w in ROLL_WINDOWS:
        cols += [f"price_roll_mean_{w}", f"price_roll_std_{w}"]
    cols += ["price_change_24", "hour_sin", "hour_cos", "dow", "hours_since_start"]
    return cols


def make_features(series: pd.DataFrame, season_start: pd.Timestamp) -> pd.DataFrame:
    price = series["median_price"].astype(float)
    volume = series["volume"].astype(float)
    listings = series["listings"].astype(float)
    f = pd.DataFrame(index=series.index)
    for lag in PRICE_LAGS:
        f[f"price_lag_{lag}"] = price.shift(lag)
    for lag in EXOG_LAGS:
        f[f"volume_lag_{lag}"] = volume.shift(lag)
        f[f"listings_lag_{lag}"] = listings.shift(lag)
    lag1 = price.shift(1)
    for w in ROLL_WINDOWS:
        f[f"price_roll_mean_{w}"] = lag1.rolling(w, min_periods=1).mean()
        f[f"price_roll_std_{w}"] = lag1.rolling(w, min_periods=2).std()
    lag24 = price.shift(24)
    f["price_change_24"] = ((lag1 - lag24) / lag24).replace([np.inf, -np.inf], np.nan)
    hour = series.index.hour.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = series.index.dayofweek.to_numpy().astype(float)
    f["hours_since_start"] = (series.index - season_start) / pd.Timedelta(hours=1)
    return f[feature_columns()]
