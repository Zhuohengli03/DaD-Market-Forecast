import time

from backend.services.catalog_sync import scan_market, sync_catalog


class _FakeClient:
    def __init__(self, items=None, market=None):
        self.items, self.market, self.calls = items or [], market or [], []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/patches/current":
            return {"body": {"id": "0.17.153.9762"}}
        raise AssertionError(path)

    def iter_pages(self, path, params, max_pages=None):
        self.calls.append((path, dict(params), max_pages))
        pages = self.items if path == "/items" else self.market
        yield from pages[: max_pages or len(pages)]


ITEMS = [[{"id": "id.item.iron_ore_1001", "name": "Iron Ore", "rarity": "common", "item_type": "Misc", "vendor_price": 5},
          {"id": "id.item.arming_sword_2001", "name": "Arming Sword", "rarity": "uncommon", "slot_type": "Weapon",
           "primary": {"weapon_damage": 30}}],
         [{"id": "id.item.gold_ore_1001", "name": "Gold Ore", "rarity": "common", "item_type": "Misc"}]]


def _listing(iid, name, ppu, ts="2026-10-01T00:00:00Z", qty=1):
    return {"id": f"{iid}-{ppu}-{ts}", "item": {"id": iid, "name": name}, "quantity": qty, "price": ppu * qty,
            "price_per_unit": ppu, "created_at": ts}


def test_sync_catalog_stores_all_raw_fields(settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    n = sync_catalog(_FakeClient(items=ITEMS), store, log=lambda s: None)
    assert n == 3
    d = store.read()
    assert d["patch"] == "0.17.153.9762" and d["items"][1]["primary"] == {"weapon_damage": 30}


def test_scan_market_aggregates_per_item(settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    market = [[_listing("id.item.iron_ore_1001", "Iron Ore", 80), _listing("id.item.iron_ore_1001", "Iron Ore", 90, "2026-10-02T00:00:00Z")],
              [_listing("id.item.iron_ore_1001", "Iron Ore", 100), {"id": "broken"}]]
    client = _FakeClient(market=market)
    seen = scan_market(client, store, since="2026-09-30T00:00:00+00:00", max_pages=10, log=lambda s: None)
    assert seen == 1
    st = store.read()["market"]["id.item.iron_ore_1001"]
    assert st["listings"] == 3 and st["min_price_per_unit"] == 80 and st["median_price_per_unit"] == 90
    assert st["last_seen"] == "2026-10-02T00:00:00Z" and st["name"] == "Iron Ore"
    path, params, max_pages = client.calls[0]
    assert path == "/market" and params["from"] == "2026-09-30T00:00:00+00:00" and max_pages == 10


def test_catalog_api_query_and_track(client, settings):
    from backend.data.catalog import CatalogStore
    store = CatalogStore(settings.catalog_path)
    sync_catalog(_FakeClient(items=ITEMS), store, log=lambda s: None)
    scan_market(_FakeClient(market=[[_listing("id.item.gold_ore_1001", "Gold Ore", 40)]]), store, since=None,
                max_pages=1, log=lambda s: None)
    r = client.get("/api/catalog", params={"q": "ore"}).json()
    assert [x["name"] for x in r["data"]] == ["Gold Ore", "Iron Ore"]      # tradable first
    assert r["meta"]["total"] == 2 and "common" in r["meta"]["facets"]["rarity"]
    assert r["data"][0]["market"]["listings"] == 1 and r["data"][1]["raw"]["vendor_price"] == 5
    only = client.get("/api/catalog", params={"tradable": "true"}).json()["data"]
    assert [x["name"] for x in only] == ["Gold Ore"]
    t = client.post("/api/items/track", json={"item_id": "id.item.arming_sword_2001"})
    assert t.status_code == 201 and t.json()["data"]["name"] == "Arming Sword"
    assert client.post("/api/items/track", json={"item_id": "id.item.arming_sword_2001"}).status_code == 409
    assert client.post("/api/items/track", json={"item_id": "id.item.nope"}).status_code == 404
    assert any(i["name"] == "Arming Sword" for i in client.get("/api/items").json()["data"])
    assert client.delete("/api/items/Arming Sword").status_code == 200


def test_catalog_jobs_need_api_key(client):
    r = client.post("/api/catalog/sync")
    assert r.status_code == 409 and "API key" in r.json()["error"]


def test_catalog_sync_job_runs_with_injected_client(client):
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: _FakeClient(items=ITEMS)
    job_id = client.post("/api/catalog/sync").json()["data"]["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert j["status"] == "succeeded", j
    assert client.get("/api/catalog/meta").json()["data"]["items"] == 3
