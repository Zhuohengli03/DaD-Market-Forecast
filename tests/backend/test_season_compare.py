import pandas as pd
from backend.services.series import season_compare


class Repo:
    def __init__(self, df): self.df = df
    def fetch_all(self, item, start, end):
        ts = pd.to_datetime(self.df.created_at, utc=True)
        m = ts >= start if end is None else (ts >= start) & (ts < end)
        return self.df[m]


def _rows(day0, days, price):
    return [{"price_per_unit": price + d, "quantity": 1, "has_sold": True, "price": price + d,
             "created_at": (pd.Timestamp(day0) + pd.Timedelta(days=d, hours=12)).isoformat()} for d in range(days)]


def test_seasons_line_up_by_day_since_trading_opened():
    df = pd.DataFrame(_rows("2026-01-10T00:00Z", 3, 10) + _rows("2026-03-01T00:00Z", 2, 20))
    seasons = [{"id": "a", "name": "A", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z", "market_open": "2026-01-10"},
               {"id": "b", "name": "B", "start": "2026-02-20T00:00:00Z", "end": None, "market_open": "2026-03-01"},
               {"id": "c", "name": "C", "start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"}]   # no data
    out = {s["id"]: s for s in season_compare(Repo(df), "X", seasons)}
    assert set(out) == {"a", "b"}
    assert out["a"]["days"] == [{"day": 0, "median": 10.0}, {"day": 1, "median": 11.0}, {"day": 2, "median": 12.0}]
    assert (out["a"]["open"], out["a"]["close"], out["a"]["high"], out["a"]["low"], out["a"]["listings"]) == (10.0, 12.0, 12.0, 10.0, 3)
    assert out["b"]["days"][0] == {"day": 0, "median": 20.0}


def test_season_compare_endpoint(client):
    d = client.get("/api/series/Test Ore/seasons").json()["data"]["seasons"]
    assert [s["id"] for s in d] == ["s2"] and d[0]["days"][0]["day"] == 0 and d[0]["listings"] > 0


def test_a_season_saved_without_a_time_zone_still_compares():
    df = pd.DataFrame(_rows("2026-01-10T00:00Z", 2, 10))
    seasons = [{"id": "a", "name": "A", "start": "2026-01-10", "end": "2026-02-01"}]      # dates only, no zone
    out = season_compare(Repo(df), "X", seasons)
    assert out[0]["days"][0] == {"day": 0, "median": 10.0}
