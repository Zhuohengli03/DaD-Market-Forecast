"""Other seasons' price paths aligned by hours-since-season-start (display reference only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceCurve:
    season_id: str
    name: str
    points: tuple[tuple[int, float], ...]


def reference_curves(current_offset_hours: int, horizon: int,
                     others: list[tuple[str, str, pd.DataFrame, pd.Timestamp]]) -> list[ReferenceCurve]:
    out: list[ReferenceCurve] = []
    for season_id, name, series, start in others:
        if series.empty:
            continue
        offset = ((series.index - start) / pd.Timedelta(hours=1)).astype(int)
        rel = offset - current_offset_hours
        prices = series["median_price"].to_numpy(dtype=float)
        mask = (rel >= 1) & (rel <= horizon) & ~np.isnan(prices)
        if not mask.any():
            continue
        pts = tuple((int(h), float(p)) for h, p in zip(rel[mask], prices[mask]))
        out.append(ReferenceCurve(season_id, name, pts))
    return out
