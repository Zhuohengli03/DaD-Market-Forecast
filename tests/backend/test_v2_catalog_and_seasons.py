import time

from backend.data.catalog import CatalogStore
from backend.services.catalog_sync import sync_catalog

H = {"X-Darker-Market": "1"}
PATCH = {"slug": "early-access-season-11", "number": "11", "kind": "season", "title": "Early Access Season #11",
         "released_at": "2026-10-09", "is_current": True}
SEASONS = [PATCH,
           {"slug": "early-access-season-10", "kind": "season", "title": "Early Access Season #10", "released_at": "2026-07-17", "is_current": False},
           {"slug": "early-access-season-9", "kind": "season", "title": "Early Access Season #9", "released_at": "2026-04-28", "is_current": False}]


class _Client:
    def get(self, path, params=None):
        if path == "/patches/current":
            return {"body": PATCH}
        if path == "/patches":
            return {"body": SEASONS, "pagination": {"next": None}}
        raise AssertionError(path)

    def iter_pages(self, path, params, max_pages=None):
        if path == "/patches":
            assert params["kind"] == "season"
            yield SEASONS
            return
        yield [{"id": "id.item.iron_ores_2001", "name": "Iron Ore", "rarity": "common", "is_tradable": True,
                "item_type": "misc", "icon_url": "https://cdn.darkerdb.com/codex/abc"},
               {"id": "id.item.quest_key_1001", "name": "Quest Key", "is_tradable": False, "item_type": "misc"}]


def test_patch_title_is_recorded(settings):
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    sync_catalog(_Client(), store, log=lambda s: None)
    assert store.meta()["patch"] == "Early Access Season #11"


def test_import_seasons_from_darkerdb(client):
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: _Client()
    preview = client.get("/api/seasons/darkerdb").json()["data"]
    assert [p["name"] for p in preview] == ["Season 9", "Season 10", "Season 11"]
    assert preview[0]["end"] == preview[1]["start"] and preview[-1]["end"] is None
    r = client.post("/api/seasons/import", json={"replace": True}, headers=H)
    assert r.status_code == 200
    seasons = client.get("/api/seasons").json()["data"]
    assert [(s["id"], s["start"][:10], (s["end"] or "")[:10]) for s in seasons] == [
        ("early-access-season-9", "2026-04-28", "2026-07-17"),
        ("early-access-season-10", "2026-07-17", "2026-10-09"),
        ("early-access-season-11", "2026-10-09", "")]


def test_import_without_replace_keeps_existing_and_skips_overlaps(client):
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: _Client()
    r = client.post("/api/seasons/import", json={"replace": False}, headers=H).json()["data"]
    assert r["skipped"] >= 1
    ids = [s["id"] for s in client.get("/api/seasons").json()["data"]]
    assert "s1" in ids
