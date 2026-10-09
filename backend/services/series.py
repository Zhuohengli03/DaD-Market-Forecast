from __future__ import annotations

import math

import pandas as pd

from src.forecast.aggregate import hourly_series, resample_series

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
    return parse_ts(s["start"]), parse_ts(s["end"])


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
