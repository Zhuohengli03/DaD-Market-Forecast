"""Pinned to response shapes observed from the live DarkerDB v2 API on 2026-10-10."""
from src.api.darkerdb_client import DarkerDBClient, listing_item_ref, normalize_listing

MARKET_ROW = {"id": 46765463, "item_id": "id.item.iron_ores_2001", "archetype": "id.item.iron_ores", "name": "Iron Ore",
              "rarity": "common", "price": 488, "price_per_unit": 81.33, "quantity": 6, "listing_state": "active",
              "has_sold": False, "created_at": "2026-10-09T09:03:20.000Z"}


class _Resp:
    def __init__(self, body):
        self.status_code, self._body, self.headers = 200, body, {}

    def json(self):
        return self._body


class _Session:
    def __init__(self, bodies):
        self.bodies, self.calls = list(bodies), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(params or {}))
        return _Resp(self.bodies.pop(0))


def test_flat_market_row_uses_name_field():
    assert listing_item_ref(MARKET_ROW) == ("id.item.iron_ores_2001", "Iron Ore")
    n = normalize_listing(MARKET_ROW, "fallback")
    assert n == {"id": "46765463", "item": "Iron Ore", "quantity": 6, "price_per_unit": 81.33, "price": 488.0,
                 "has_sold": False, "created_at": "2026-10-09T09:03:20Z"}


def test_page_based_next_url_is_followed():
    s = _Session([
        {"body": [MARKET_ROW], "pagination": {"page": 1, "next": "https://api.darkerdb.com/v2/market?item_id=x&limit=50&page=2&from=2026-10-01"}},
        {"body": [MARKET_ROW], "pagination": {"page": 2, "next": None}}])
    c = DarkerDBClient("k", session=s, min_interval=0)
    pages = list(c.iter_market("x", since="2026-10-01"))
    assert len(pages) == 2
    assert s.calls[1]["page"] == "2" and "cursor" not in s.calls[1] and s.calls[1]["item_id"] == "x"


def test_opaque_cursor_still_supported():
    s = _Session([{"body": [{"id": "a"}], "pagination": {"next": "eyJ2Ijoi"}}, {"body": [], "pagination": {"next": None}}])
    list(DarkerDBClient("k", session=s, min_interval=0).iter_pages("/items", {"limit": 200}))
    assert s.calls[1] == {"limit": 200, "cursor": "eyJ2Ijoi"}
