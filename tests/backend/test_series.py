def test_items_merges_config_and_stats(client):
    data = client.get("/api/items").json()["data"]
    assert data[0]["name"] == "Test Ore" and data[0]["rows"] == 1000 and data[0]["csv"] == "test_ore.csv"


def test_series_by_season(client):
    r = client.get("/api/series/Test Ore", params={"season_id": "s2"})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["resolution"] == "1h" and len(d["points"]) == 200
    p = d["points"][0]
    assert set(p) >= {"ts", "median_price", "p25", "p75", "listings", "sold", "volume"}


def test_series_resolution_and_range(client):
    r = client.get("/api/series/Test Ore", params={"start": "2026-01-20T00:00:00Z", "end": "2026-01-22T00:00:00Z", "resolution": "4h"})
    assert len(r.json()["data"]["points"]) == 12
    assert client.get("/api/series/Test Ore", params={"resolution": "2h"}).status_code == 422


def test_series_unknown_item_and_season(client):
    assert client.get("/api/series/Ghost").status_code == 404
    assert client.get("/api/series/Test Ore", params={"season_id": "nope"}).status_code == 404
    assert client.get("/api/series/bad;name").status_code == 422
