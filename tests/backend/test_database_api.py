"""The local database page: look at every stored listing, delete some, never edit."""
import json
import sqlite3

H = {"X-Darker-Market": "1"}


def _add_untracked(settings):
    con = sqlite3.connect(settings.sqlite_path)
    con.execute("INSERT INTO listings VALUES ('old_ore','x1','Old Ore',1,5.0,5.0,1,'2025-01-01T00:00:00Z','2025-01-01T00:00:00Z')")
    con.commit(); con.close()


def test_lists_every_stored_item_including_untracked_ones(client, settings):
    _add_untracked(settings)
    items = {i["key"]: i for i in client.get("/api/db/items").json()["data"]}
    assert items["test_ore"]["rows"] == 1000 and items["test_ore"]["name"] == "Test Ore" and items["test_ore"]["tracked"] is True
    assert items["old_ore"]["rows"] == 1 and items["old_ore"]["tracked"] is False and items["old_ore"]["name"] == "Old Ore"
    assert client.get("/api/db/items").json()["meta"]["source"] == "sqlite"


def test_pages_and_filters_rows(client):
    d = client.get("/api/db/rows", params={"key": "test_ore", "page_size": 10}).json()
    assert len(d["data"]) == 10 and d["meta"]["total"] == 1000
    assert d["data"][0]["created_at"] >= d["data"][-1]["created_at"]           # newest first
    assert {"id", "created_at", "price_per_unit", "quantity", "price", "has_sold"} <= set(d["data"][0])
    some = client.get("/api/db/rows", params={"key": "test_ore", "start": "2026-01-20T00:00:00Z", "end": "2026-01-20T02:00:00Z"}).json()
    assert 0 < some["meta"]["total"] <= 10
    assert client.get("/api/db/rows", params={"key": "nope_ore"}).json()["meta"]["total"] == 0


def test_deletes_chosen_rows_a_time_range_or_everything(client, settings):
    page = client.get("/api/db/rows", params={"key": "test_ore", "page_size": 3}).json()["data"]
    r = client.post("/api/db/delete", json={"key": "test_ore", "ids": [p["id"] for p in page]}, headers=H)
    assert r.status_code == 200 and r.json()["data"]["deleted"] == 3
    r = client.post("/api/db/delete", json={"key": "test_ore", "start": "2026-01-20T00:00:00Z", "end": "2026-01-20T02:00:00Z"}, headers=H)
    assert r.json()["data"]["deleted"] > 0
    left = client.get("/api/db/rows", params={"key": "test_ore"}).json()["meta"]["total"]
    assert left == 1000 - 3 - r.json()["data"]["deleted"]
    assert next(i for i in client.get("/api/items").json()["data"] if i["name"] == "Test Ore")["rows"] == left   # stats refreshed
    # nothing chosen is refused; the whole item needs an explicit "all"
    assert client.post("/api/db/delete", json={"key": "test_ore"}, headers=H).status_code == 422
    r = client.post("/api/db/delete", json={"key": "test_ore", "all": True}, headers=H)
    assert r.json()["data"]["deleted"] == left
    assert client.get("/api/db/rows", params={"key": "test_ore"}).json()["meta"]["total"] == 0


def test_deleting_forgets_how_far_seasons_were_collected(client, settings):
    cov = settings.data_dir / "coverage" / "test_ore.json"
    cov.parent.mkdir(parents=True, exist_ok=True)
    cov.write_text(json.dumps({"s1": "2026-01-01T00:00:00Z"}))
    client.post("/api/db/delete", json={"key": "test_ore", "start": "2026-01-20T00:00:00Z", "end": "2026-01-20T01:00:00Z"}, headers=H)
    assert not cov.exists()                          # the next collection fills the hole again


def test_rows_cannot_be_edited(client):
    assert client.put("/api/db/rows", json={}, headers=H).status_code == 405
    assert client.patch("/api/db/rows", json={}, headers=H).status_code == 405


def test_items_carry_their_chinese_names(client, settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    CatalogStore(settings.catalog_path).save_items([
        {"id": "id.item.old_ores_1001", "archetype": "id.item.old_ores", "name": "Old Ore", "name_zh": "旧矿石", "rarity": "common",
         "item_type": "misc", "num_secondary_attributes": 0, "is_tradable": True}], "Early Access Season #11")
    _add_untracked(settings)
    items = {i["key"]: i for i in client.get("/api/db/items").json()["data"]}
    assert items["old_ore"]["name_zh"] == "旧矿石"                # untracked: matched by name in the catalog
    assert items["test_ore"]["name_zh"] is None                  # not in the catalog: English stays
