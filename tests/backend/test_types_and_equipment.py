import json

from backend.data.catalog import CatalogStore
from backend.services.catalog_sync import check_tradability
from backend.services.equipment import build_market_params, daily_stats

H = {"X-Darker-Market": "1"}


def V(arch, n, rarity, sec, name, item_type="weapon"):
    return {"id": f"id.item.{arch}_{n}", "archetype": f"id.item.{arch}", "name": name, "rarity": rarity, "item_type": item_type,
            "num_primary_attributes": 2, "num_secondary_attributes": sec, "is_tradable": True,
            "icon_url": f"https://cdn/{arch}_{n}", "gear_score": 10}


VARIANTS = [V("arming_sword", 2001, "common", 0, "Arming Sword"), V("arming_sword", 6001, "legendary", 116, "Arming Sword"),
            {"id": "id.item.iron_ores_2001", "archetype": "id.item.iron_ores", "name": "Iron Ore", "rarity": "common",
             "item_type": "misc", "num_primary_attributes": 0, "num_secondary_attributes": 0, "is_tradable": True},
            {"id": "id.item.bandage_1001", "archetype": "id.item.bandage", "name": "Bandage", "rarity": "poor", "item_type": "utility", "num_secondary_attributes": 0, "is_tradable": True},
            {"id": "id.item.bandage_2001", "archetype": "id.item.bandage", "name": "Bandage", "rarity": "common", "item_type": "utility", "num_secondary_attributes": 0, "is_tradable": True},
            {"id": "id.item.gold_coins", "archetype": "id.item.gold_coins", "name": "Gold Coin", "rarity": "unique", "item_type": "misc", "num_secondary_attributes": 0, "is_tradable": True}]


def _seed(settings, totals=None):
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    store.save_items(VARIANTS, "Early Access Season #11")
    if totals is not None:
        store.save_trade_totals(totals, complete=True)
    return store


def test_types_are_grouped_and_classified(settings):
    store = _seed(settings)
    rows, total, facets = store.query()
    names = {r["name"]: r for r in rows}
    assert total == 4 and set(names) == {"Arming Sword", "Iron Ore", "Bandage", "Gold Coin"}
    sword = names["Arming Sword"]
    assert sword["archetype"] == "id.item.arming_sword" and sword["has_stats"] is True
    assert [v["rarity"] for v in sword["variants"]] == ["common", "legendary"]
    assert names["Bandage"]["has_stats"] is False and len(names["Bandage"]["variants"]) == 2
    assert "weapon" in facets["item_type"]


def test_untradable_types_hidden_after_check(settings):
    store = _seed(settings, totals={"id.item.arming_sword": 95000, "id.item.iron_ores": 58000, "id.item.bandage": 10, "id.item.gold_coins": 0})
    rows, total, _ = store.query()
    assert {r["name"] for r in rows} == {"Arming Sword", "Iron Ore", "Bandage"}
    assert {r["name"]: r["listings_total"] for r in rows}["Iron Ore"] == 58000
    assert store.meta()["trade_check_complete"] is True and store.meta()["types"] == 3


def test_has_stats_filter(settings):
    store = _seed(settings)
    rows, _, _ = store.query(has_stats=False)
    assert {r["name"] for r in rows} == {"Iron Ore", "Bandage", "Gold Coin"}


def test_check_tradability_is_resumable(settings):
    store = _seed(settings)
    calls = []

    class C:
        def get(self, path, params=None):
            calls.append(params["archetype"])
            return {"body": [], "pagination": {"total": 0 if params["archetype"] == "id.item.gold_coins" else 7}}
    check_tradability(C(), store, log=lambda s: None)
    assert len(calls) == 4 and store.read()["trade_totals"]["id.item.gold_coins"] == 0
    calls.clear()
    check_tradability(C(), store, log=lambda s: None)
    assert calls == []                                   # already checked; nothing re-requested
    check_tradability(C(), store, log=lambda s: None, force=True)
    assert len(calls) == 4


def test_track_rejects_types_with_stats(client, settings):
    _seed(settings)
    r = client.post("/api/items/track", json={"item_id": "id.item.arming_sword_6001"}, headers=H)
    assert r.status_code == 422 and "装备行情" in r.json()["error"]
    assert client.post("/api/items/track", json={"item_id": "id.item.bandage_2001"}, headers=H).status_code == 201
    names = [i["name"] for i in client.get("/api/items").json()["data"]]
    assert "Bandage (common)" in names or "Bandage" in names


def test_catalog_api_returns_types(client, settings):
    _seed(settings, totals={"id.item.arming_sword": 1, "id.item.iron_ores": 1, "id.item.bandage": 1, "id.item.gold_coins": 0})
    d = client.get("/api/catalog", params={"q": "sword"}).json()
    assert d["data"][0]["name"] == "Arming Sword" and d["meta"]["trade_check_complete"] is True


# ---- equipment market ------------------------------------------------------
def test_market_params_from_filters():
    p = build_market_params("id.item.arming_sword", "legendary", [
        {"kind": "secondary", "name": "armor_penetration", "min": 2, "max": 5},
        {"kind": "secondary", "name": "headshot_damage_bonus", "min": 3, "max": None},
        {"kind": "primary", "name": "weapon_damage", "min": None, "max": 34}], sold_only=True, since="2026-10-09T00:00:00+00:00")
    assert p == {"archetype": "id.item.arming_sword", "rarity": "legendary", "has_sold": "true", "limit": 50,
                 "sort": "created_at:desc", "from": "2026-10-09T00:00:00+00:00",
                 "secondary[armor_penetration]": "2:5", "secondary[headshot_damage_bonus]": ">=3", "primary[weapon_damage]": "<=34"}


def test_daily_stats():
    rows = [{"created_at": "2026-10-09T01:00:00Z", "price_per_unit": 100, "has_sold": True},
            {"created_at": "2026-10-09T05:00:00Z", "price_per_unit": 300, "has_sold": False},
            {"created_at": "2026-10-09T09:00:00Z", "price_per_unit": 200, "has_sold": True},
            {"created_at": "2026-10-10T01:00:00Z", "price_per_unit": 50, "has_sold": True}]
    d = daily_stats(rows)
    assert d == [{"day": "2026-10-09", "listings": 3, "sold": 2, "median": 200, "low": 100, "high": 300, "sold_median": 150},
                 {"day": "2026-10-10", "listings": 1, "sold": 1, "median": 50, "low": 50, "high": 50, "sold_median": 50}]


class _MarketClient:
    def __init__(self):
        self.params = []

    def get(self, path, params=None):
        if path == "/attributes":
            return {"body": [{"id": "id.attribute.armor_penetration", "name": "Armor Penetration", "attribute_group": "secondary", "is_percentage": True}],
                    "pagination": {"next": None}}
        raise AssertionError(path)

    def iter_pages(self, path, params, max_pages=None):
        self.params.append(dict(params))
        if path == "/attributes":
            yield [{"id": "id.attribute.armor_penetration", "name": "Armor Penetration", "is_percentage": True},
                   {"id": "id.attribute.weapon_damage", "name": "Weapon Damage", "is_percentage": False}]
            return
        yield [{"id": 1, "item_id": "id.item.arming_sword_6001", "name": "Arming Sword", "rarity": "legendary", "quantity": 1,
                "price": 2333, "price_per_unit": 2333, "has_sold": True, "listing_state": "missing", "created_at": "2026-10-09T10:00:00.000Z",
                "attributes": {"primary_weapon_damage": 32, "secondary_armor_penetration": 2.9}},
               {"id": 2, "item_id": "id.item.arming_sword_6001", "name": "Arming Sword", "rarity": "legendary", "quantity": 1,
                "price": 3333, "price_per_unit": 3333, "has_sold": False, "listing_state": "active", "created_at": "2026-10-09T11:00:00.000Z",
                "attributes": {"primary_weapon_damage": 31, "secondary_armor_penetration": 4.1}}]


def test_equipment_query_endpoint(client, settings):
    _seed(settings)
    fake = _MarketClient()
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: fake
    body = {"archetype": "id.item.arming_sword", "rarity": "legendary", "sold_only": False, "max_pages": 2,
            "filters": [{"kind": "secondary", "name": "armor_penetration", "min": 2, "max": 5}]}
    d = client.post("/api/equipment/query", json=body, headers=H).json()["data"]
    assert len(d["listings"]) == 2 and d["listings"][0]["attributes"]["secondary_armor_penetration"] == 2.9
    assert d["daily"][0]["listings"] == 2 and d["daily"][0]["sold"] == 1
    market_call = [p for p in fake.params if "archetype" in p][0]
    assert market_call["secondary[armor_penetration]"] == "2:5" and market_call["rarity"] == "legendary"


def test_equipment_query_validation(client, settings):
    _seed(settings)
    client.app.state.settings.darker_market_api_key = "k"
    bad = {"archetype": "id.item.arming_sword", "filters": [{"kind": "tertiary", "name": "x"}]}
    assert client.post("/api/equipment/query", json=bad, headers=H).status_code == 422
    bad2 = {"archetype": "id.item.arming_sword", "filters": [{"kind": "secondary", "name": "luck", "min": 5, "max": 1}]}
    assert client.post("/api/equipment/query", json=bad2, headers=H).status_code == 422
    assert client.post("/api/equipment/query", json={"archetype": "id.item.iron_ores"}, headers=H).status_code == 422


def test_equipment_stats_endpoint(client, settings):
    _seed(settings)
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: _MarketClient()
    d = client.get("/api/equipment/id.item.arming_sword/stats").json()["data"]
    keyed = {s["key"]: s for s in d["stats"]}
    assert keyed["secondary_armor_penetration"]["label"] == "Armor Penetration" and keyed["secondary_armor_penetration"]["percent"] is True
    assert keyed["secondary_armor_penetration"]["min"] == 2.9 and keyed["secondary_armor_penetration"]["max"] == 4.1
    assert keyed["primary_weapon_damage"]["kind"] == "primary"
    assert d["rarities"] == ["common", "legendary"]


def test_stat_labels_include_chinese(client, settings):
    _seed(settings)

    class Zh(_MarketClient):
        def iter_pages(self, path, params, max_pages=None):
            if path == "/attributes" and params.get("locale") == "zh-Hans":
                yield [{"id": "id.attribute.armor_penetration", "name": "护甲穿透", "is_percentage": True}]
                return
            yield from super().iter_pages(path, params, max_pages)
    from backend.services import equipment
    equipment._attr_cache.clear()
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.client_factory = lambda key: Zh()
    stats = {s["key"]: s for s in client.get("/api/equipment/id.item.arming_sword/stats").json()["data"]["stats"]}
    assert stats["secondary_armor_penetration"]["label_zh"] == "护甲穿透"
