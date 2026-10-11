"""Wipe dates alone must not classify stale pre-market listings as a new economy."""
import pandas as pd

from backend.services.collection import collect_range
from src.forecast.pipeline import ForecastRequest, InsufficientDataError, run_forecast
from src.season_calendar import market_start, with_market_opening
import pytest


def set_opening(client, date="2026-02-01"):
    s = client.app.state.seasons.get("s2")
    r = client.put("/api/seasons/s2", json={**s, "market_open": date})
    assert r.status_code == 200


def test_ssf_records_excluded_from_season_but_retained_for_custom_query(client):
    # Fixture has 200 hours starting Jan 20: all precede the Feb 1 market opening.
    set_opening(client)
    r = client.get("/api/series/Test Ore", params={"season_id": "s2"}).json()["data"]
    assert r["start"] == "2026-02-01T00:00:00+00:00"
    assert r["points"] == []
    listings = client.get("/api/listings/Test Ore", params={"season_id": "s2"}).json()
    assert listings["meta"]["total"] == 0
    raw = client.get("/api/series/Test Ore").json()["data"]
    assert len(raw["points"]) == 200  # preserve the historical evidence
    assert client.get("/api/items").json()["data"][0]["season_rows"] == 0
    v = client.get("/api/overview").json()["data"]["groups"][0]["variants"][0]
    assert v["latest"] is None and v["season_rows"] == 0


def test_custom_range_cannot_bypass_selected_season_or_market_opening(client):
    set_opening(client, "2026-01-22")
    params = {"season_id": "s2", "start": "2025-01-01", "end": "2026-01-23"}
    data = client.get("/api/series/Test Ore", params=params).json()["data"]
    assert len(data["points"]) == 24
    assert data["points"][0]["ts"].startswith("2026-01-22")
    assert client.get("/api/series/Test Ore", params={**params, "end": "2026-01-21"}).status_code == 422


def test_collection_and_forecast_use_same_market_boundary(client):
    set_opening(client)
    st = client.app.state
    since, until = collect_range(st, "Test Ore", "s2")
    assert pd.Timestamp(since) == pd.Timestamp("2026-02-01", tz="UTC") and until is None
    # Existing pre-opening rows must not train a seemingly valid forecast.
    with pytest.raises(InsufficientDataError):
        run_forecast(ForecastRequest("Test Ore", "s2", 24, ("naive",)), st.repo.fetch_all, st.seasons.to_pipeline())
    assert st.seasons.get("s2")["start"].startswith("2026-01-20")


def test_opening_validation_and_preservation(client):
    s = client.app.state.seasons.get("s2")
    for invalid in ("2026-01-19", "2026-02-30", "garbage"):
        assert client.put("/api/seasons/s2", json={**s, "market_open": invalid}).status_code == 422
    set_opening(client)
    assert client.put("/api/seasons/s2", json={**s, "name": "Renamed"}).status_code == 200
    assert client.app.state.seasons.get("s2")["market_open"] == "2026-02-01"
    client.app.state.seasons.import_records([s], replace=True)
    assert client.app.state.seasons.get("s2")["market_open"] == "2026-02-01"


def test_known_opening_enriches_old_records_without_changing_wipe():
    old = {"id": "early-access-season-9", "start": "2026-04-28T00:00:00Z"}
    assert with_market_opening(old)["market_open"] == "2026-05-14"
    assert market_start(old).startswith("2026-05-14")
    assert old["start"] == "2026-04-28T00:00:00Z" and "market_open" not in old
    assert market_start({**old, "market_open": None}) == old["start"]
    assert "market_open" not in with_market_opening({**old, "id": "future-season"})


def test_import_keeps_exact_timestamp_if_provider_supplies_one(client):
    from backend.api.seasons import seasons_from_darkerdb
    class Patches:
        def iter_pages(self, *args):
            yield [{"slug": "exact", "number": "99", "kind": "season", "released_at": "2026-10-09T13:30:00+02:00"}]
    season = seasons_from_darkerdb(client.app.state, Patches())[0]
    assert season["start"] == "2026-10-09T11:30:00+00:00"


@pytest.mark.parametrize("number,start,opening,end", [
    (6, "2025-04-10", "2025-04-24", "2025-09-24"),
    (7, "2025-09-24", "2025-10-07", "2026-01-29"),
    (8, "2026-01-29", "2026-02-20", "2026-04-28"),
    (9, "2026-04-28", "2026-05-14", "2026-07-17"),
    (10, "2026-07-17", "2026-07-30", "2026-10-09"),
])
def test_historical_openings_apply_to_store_queries_and_reimport(client, number, start, opening, end):
    store = client.app.state.seasons
    record = {"id": f"early-access-season-{number}", "name": f"Season {number}",
              "start": start, "end": end}
    assert store.import_records([record], replace=True) == (1, 0)
    season = store.get(record["id"])
    assert season["start"][:10] == start and season["market_open"] == opening
    data = client.get("/api/series/Test Ore", params={"season_id": record["id"]}).json()["data"]
    assert data["start"][:10] == opening and data["end"][:10] == end
    pipeline_season = store.to_pipeline()[0]
    assert pipeline_season.start == pd.Timestamp(start, tz="UTC")
    assert pipeline_season.market_start == pd.Timestamp(opening, tz="UTC")
    # An explicit manual clearing survives future imports of the patch calendar.
    r = client.put(f"/api/seasons/{record['id']}", json={**season, "market_open": None})
    assert r.status_code == 200
    store.import_records([record], replace=True)
    assert store.get(record["id"])["market_open"] is None


def test_unknown_or_manually_shifted_seasons_do_not_get_guessed_openings():
    for number, start in [(4, "2024-09-05"), (5, "2024-12-18"), (10, "2026-08-01")]:
        season = {"id": f"early-access-season-{number}", "start": start}
        assert with_market_opening(season) == season
        assert market_start(season) == start
