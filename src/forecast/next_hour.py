"""Where will next hour's individual sales land? A range for buying and selling, not a direction call.

Range = the latest hour's median sale price x quantiles of (sale price / the median of the hour before it), taken
from the last CALIB_DAYS of sales only, so it widens when the market gets rougher. Its hit rate over the last
CHECK_DAYS is measured the same way (each hour judged with a range built from the days before it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import InsufficientDataError

CALIB_DAYS = 3
CHECK_DAYS = 7
MIN_RATIOS = 100
QUANTILES = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}


def _ratios(listings: pd.DataFrame) -> pd.DataFrame:
    """Each sale with its hour and its price as a share of the previous hour's median (from earlier hours only)."""
    sold = listings[listings["has_sold"].astype(bool)]
    ts = pd.to_datetime(sold["created_at"], utc=True)
    d = pd.DataFrame({"hour": ts.dt.floor("h"), "p": sold["price_per_unit"].astype(float).to_numpy(),
                      "q": sold["quantity"].astype(float).to_numpy()}, index=sold.index)
    med = d.groupby("hour")["p"].median().asfreq("h").ffill()
    d["level"] = d["hour"].map(med.shift(1))
    return d.dropna(subset=["level"]).assign(r=lambda x: x["p"] / x["level"]), med


def _bounds(r: pd.Series) -> dict[str, float]:
    return {k: float(r.quantile(v)) for k, v in QUANTILES.items()}


def _check(d: pd.DataFrame, end: pd.Timestamp) -> dict:
    """Quantity-weighted share of sales inside P25-P75 and P10-P90 over the last CHECK_DAYS, each hour judged
    with a range calibrated on the CALIB_DAYS before it."""
    mid = wide = total = 0.0
    hours = 0
    for h in pd.date_range(end - pd.Timedelta(days=CHECK_DAYS), end - pd.Timedelta(hours=1), freq="h"):
        now = d[d["hour"] == h]
        past = d[(d["hour"] < h) & (d["hour"] >= h - pd.Timedelta(days=CALIB_DAYS))]["r"]
        if now.empty or len(past) < MIN_RATIOS:
            continue
        b = _bounds(past)
        mid += now["q"][(now["r"] >= b["p25"]) & (now["r"] <= b["p75"])].sum()
        wide += now["q"][(now["r"] >= b["p10"]) & (now["r"] <= b["p90"])].sum()
        total += now["q"].sum()
        hours += 1
    return {"mid": float(mid / total) if total else None, "wide": float(wide / total) if total else None, "hours": hours}


def next_hour_range(listings: pd.DataFrame, now: pd.Timestamp | None = None, check: bool = True) -> dict:
    """`check=False` skips measuring the hit rate (the slow part), for the overview cards."""
    d, med = _ratios(listings)
    if med.empty:
        raise InsufficientDataError("还没有成交记录，先采集这个道具的数据")
    last_hour = med.index[-1]
    next_hour = last_hour + pd.Timedelta(hours=1)
    recent = d[d["hour"] >= next_hour - pd.Timedelta(days=CALIB_DAYS)]["r"]
    if len(recent) < MIN_RATIOS:
        raise InsufficientDataError(f"最近 {CALIB_DAYS} 天只有 {len(recent)} 条成交，至少需要 {MIN_RATIOS} 条才能估计价格范围")
    level = float(med.iloc[-1])
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    window = d[d["hour"] >= next_hour - pd.Timedelta(days=CHECK_DAYS + CALIB_DAYS)]
    return {
        "level": level, "level_hour": last_hour.isoformat(), "next_hour": next_hour.isoformat(),
        "prices": {k: level * v for k, v in _bounds(recent).items()},
        "check": _check(window, next_hour) if check else {"mid": None, "wide": None, "hours": 0}, "calib_days": CALIB_DAYS, "check_days": CHECK_DAYS,
        "sales_used": int(len(recent)),
        "stale_hours": max(0.0, float((now - next_hour) / pd.Timedelta(hours=1))),
    }
