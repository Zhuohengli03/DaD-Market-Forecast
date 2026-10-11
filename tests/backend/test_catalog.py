import time

from backend.services.catalog_sync import sync_catalog


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


def test_sync_catalog_stores_all_raw_fields(settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    n = sync_catalog(_FakeClient(items=ITEMS), store, log=lambda s: None)
    assert n == 3
    d = store.read()
    assert d["patch"] == "0.17.153.9762" and d["items"][1]["primary"] == {"weapon_damage": 30}


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


def test_types_are_grouped_once_until_the_catalog_changes(settings, monkeypatch):
    from backend.data import catalog as cat
    from tests.backend.test_types_and_equipment import VARIANTS, _seed
    store = _seed(settings)
    calls = []
    real = cat.type_key
    monkeypatch.setattr(cat, "type_key", lambda raw: (calls.append(1), real(raw))[1])
    store.types(); n = len(calls)
    assert n == len(VARIANTS)
    store.types(); store.variant("id.item.bandage_1001"); store.variant("id.item.iron_ores_2001")
    assert len(calls) == n                                         # remembered: no regrouping per lookup
    assert store.variant("id.item.bandage_1001")[0]["rarity"] == "poor" and store.variant("id.item.nope") is None
    store.save_items(VARIANTS[:2], "p")                             # the catalog changed: grouped again
    assert set(store.types()) == {"id.item.arming_sword"} and store.variant("id.item.bandage_1001") is None
