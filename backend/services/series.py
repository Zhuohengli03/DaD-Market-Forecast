from __future__ import annotations

import math

import pandas as pd

from src.forecast.aggregate import hourly_series, resample_series
from src.season_calendar import market_start

RESOLUTIONS = ("1h", "4h", "1d")


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def parse_ts(value: str | None) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError(f"invalid timestamp {value!r}")
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def season_range(seasons, season_id: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    s = seasons.get(season_id)  # KeyError if missing
    return parse_ts(market_start(s)), parse_ts(s["end"])


def get_series(repo, item: str, start, end, resolution: str) -> dict:
    df = repo.fetch_all(item, start, end)
    series = resample_series(hourly_series(df), resolution)
    points = [{
        "ts": ts.isoformat(),
        "median_price": _num(r.get("median_price")), "min_price": _num(r.get("min_price")),
        "max_price": _num(r.get("max_price")), "p25": _num(r.get("p25")), "p75": _num(r.get("p75")),
        "listings": int(r.get("listings", 0) or 0), "sold": int(r.get("sold", 0) or 0),
        "volume": int(r.get("volume", 0) or 0),
    } for ts, r in series.iterrows()]
    return {"item": item, "resolution": resolution,
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None, "points": points}


def season_compare(repo, item: str, seasons: list[dict]) -> list[dict]:
    """Each season's daily median sale price, day 0 = the day trading opened, plus open/high/low/close/mean."""
    from src.season_calendar import market_start
    out = []
    for s in sorted(seasons, key=lambda x: x["start"]):
        start = parse_ts(market_start(s))          # seasons saved with dates only are read as UTC
        end = parse_ts(s.get("end"))
        try:
            df = repo.fetch_all(item, start, end)
        except KeyError:
            continue
        sold = df[df["has_sold"].astype(bool)] if len(df) else df
        if sold.empty:
            continue
        ts = pd.to_datetime(sold["created_at"], utc=True)
        daily = sold["price_per_unit"].astype(float).groupby(ts.dt.floor("D")).median()
        day0 = start.floor("D")
        days = [{"day": int((d - day0) / pd.Timedelta(days=1)), "median": float(v)} for d, v in daily.items()]
        vals = daily.to_numpy()
        out.append({"id": s["id"], "name": s["name"], "market_start": start.isoformat(), "days": days,
                    "open": float(vals[0]), "close": float(vals[-1]), "high": float(vals.max()),
                    "low": float(vals.min()), "mean": float(vals.mean()), "listings": int(len(sold))})
    return out
