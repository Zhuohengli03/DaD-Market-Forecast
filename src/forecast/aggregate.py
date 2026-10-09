"""Listings -> hourly UTC series. The hourly median is the forecast target."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_COLUMNS: tuple[str, ...] = (
    "median_price", "min_price", "max_price", "p25", "p75", "listings", "sold", "volume",
)
_TRUE_STRINGS = {"true", "1", "t", "yes"}
_RESOLUTIONS = {"1h": None, "4h": "4h", "1d": "1D"}


def _empty_series() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in SERIES_COLUMNS}, index=idx)


def _coerce_sold(col: pd.Series) -> pd.Series:
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)


def hourly_series(listings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw listings into a continuous hourly series (NaN price on empty hours)."""
    if listings.empty:
        return _empty_series()
    df = pd.DataFrame({
        "created_at": pd.to_datetime(listings["created_at"], utc=True, errors="coerce"),
        "price_per_unit": pd.to_numeric(listings["price_per_unit"], errors="coerce"),
        "quantity": pd.to_numeric(listings["quantity"], errors="coerce"),
        "has_sold": _coerce_sold(listings["has_sold"]),
    }).dropna(subset=["created_at", "price_per_unit", "quantity"])
    if df.empty:
        return _empty_series()
    df["ts"] = df["created_at"].dt.floor("h")
    g = df.groupby("ts")
    out = pd.DataFrame({
        "median_price": g["price_per_unit"].median(),
        "min_price": g["price_per_unit"].min(),
        "max_price": g["price_per_unit"].max(),
        "p25": g["price_per_unit"].quantile(0.25),
        "p75": g["price_per_unit"].quantile(0.75),
        "listings": g.size(),
        "sold": g["has_sold"].sum(),
        "volume": g["quantity"].sum(),
    })
    full = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC", name="ts")
    out = out.reindex(full)
    for c in ("listings", "sold", "volume"):
        out[c] = out[c].fillna(0).astype(int)
    return out[list(SERIES_COLUMNS)]


def prepare_target(series: pd.DataFrame, max_gap_hours: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill gaps <= max_gap_hours; cut at longer gaps; return last contiguous segment."""
    warnings: list[str] = []
    s = series.copy()
    missing = s["median_price"].isna()
    if missing.all():
        return s.iloc[0:0], ["series has no prices"]
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    long_gap = missing & (run_len > max_gap_hours)
    if long_gap.any():
        last_gap_end = int(np.flatnonzero(long_gap.to_numpy())[-1])
        gap_hours = int(run_len.iloc[last_gap_end])
        warnings.append(
            f"gap of {gap_hours} hours ending at {s.index[last_gap_end].isoformat()} exceeds "
            f"{max_gap_hours}h; using only the {len(s) - last_gap_end - 1} hours after it"
        )
        s = s.iloc[last_gap_end + 1:]
    s = s.loc[s["median_price"].first_valid_index():]
    for c in ("median_price", "min_price", "max_price", "p25", "p75"):
        if c in s.columns:
            s[c] = s[c].ffill()
    return s, warnings


def resample_series(series: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Display-only resampling. 1h returns the input unchanged."""
    if resolution not in _RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(_RESOLUTIONS)}, got {resolution!r}")
    rule = _RESOLUTIONS[resolution]
    if rule is None:
        return series
    agg = {"median_price": "median", "min_price": "min", "max_price": "max", "p25": "mean",
           "p75": "mean", "listings": "sum", "sold": "sum", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in series.columns}
    return series.resample(rule).agg(agg)
