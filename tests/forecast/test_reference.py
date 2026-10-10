import numpy as np
import pandas as pd

from src.forecast.reference import reference_curves


def _series(start: str, hours: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="ts")
    return pd.DataFrame({"median_price": np.arange(hours, dtype=float)}, index=idx)


def test_reference_aligns_by_offset():
    s2 = _series("2025-10-06T00:00:00Z", 200)
    s3 = _series("2026-03-23T00:00:00Z", 50)            # too short to overlap
    curves = reference_curves(current_offset_hours=100, horizon=24, others=[
        ("s2", "Season 2", s2, pd.Timestamp("2025-10-06T00:00:00Z")),
        ("s3", "Season 3", s3, pd.Timestamp("2026-03-23T00:00:00Z")),
    ])
    assert [c.season_id for c in curves] == ["s2"]
    pts = dict(curves[0].points)
    assert pts[1] == 101.0 and pts[24] == 124.0 and len(pts) == 24


def test_reference_skips_nan_points():
    s = _series("2025-10-06T00:00:00Z", 200)
    s.iloc[105, 0] = np.nan
    curves = reference_curves(100, 10, [("s", "S", s, pd.Timestamp("2025-10-06T00:00:00Z"))])
    assert 5 not in dict(curves[0].points) and len(curves[0].points) == 9
