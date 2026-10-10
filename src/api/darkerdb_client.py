"""Minimal client for the DarkerDB v2 API (https://darkerdb.com/documentation).

Differences from the retired v1 API that the old collectors used:
  * base path /v2, auth header X-API-Key (v1 used Authorization)
  * items are filtered by item_id (e.g. id.item.iron_ore_1001), not by display name
  * cursor pagination (pagination.next), limit <= 50, sort=field:asc|desc
  * live endpoints are rate limited to 60 requests / 60 s; 429 carries Retry-After
"""
from __future__ import annotations

import time
from typing import Callable, Iterator

import requests

BASE_URL = "https://api.darkerdb.com/v2"
MARKET_MAX_LIMIT = 50
LIVE_MIN_INTERVAL = 1.1  # seconds between calls -> stays under 60 req/min


class DarkerDBError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"HTTP {code}: {message}")
        self.code = code


def _error_message(payload: dict) -> str:
    flash = payload.get("flash") or {}
    errors = flash.get("errors") or payload.get("errors") or []
    msgs = [e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in errors]
    return "; ".join(msgs) or payload.get("status", "request failed")


def _to_float(v) -> float:
    return float(v)


def canonical_ts(value) -> str:
    """UTC second precision with Z, the format v1 stored, so dedupe keys stay stable across API versions."""
    from datetime import datetime, timezone
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def listing_item_ref(row: dict) -> tuple[str | None, str | None]:
    """(item_id, display name) of a market row, whether `item` is a string or a nested object."""
    item = row.get("item")
    if isinstance(item, dict):
        return (item.get("id") or row.get("item_id")), (item.get("name") or row.get("name"))
    # live v2 rows are flat: item_id + name
    return row.get("item_id"), (item or row.get("name") or row.get("item_name"))


def normalize_listing(row: dict, fallback_item_name: str) -> dict:
    """Map a v2 market row onto the legacy DB record shape used by DarkerMarketDB."""
    _, ref_name = listing_item_ref(row)
    item_name = ref_name or fallback_item_name
    quantity = int(row["quantity"])
    price = _to_float(row["price"])
    ppu = row.get("price_per_unit")
    price_per_unit = _to_float(ppu) if ppu is not None else (price / quantity if quantity else price)
    has_sold = row.get("has_sold")
    if has_sold is None:
        has_sold = row.get("listing_state") == "sold"
    return {
        "id": str(row["id"]),
        "item": str(item_name),
        "quantity": quantity,
        "price_per_unit": price_per_unit,
        "price": price,
        "has_sold": bool(has_sold),
        "created_at": canonical_ts(row["created_at"]),
    }


class DarkerDBClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, session=None,
                 min_interval: float = LIVE_MIN_INTERVAL, sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 3, timeout: float = 30.0):
        if not api_key:
            raise DarkerDBError(401, "DARKER_MARKET_API_KEY is empty; create a key at https://darkerdb.com/dashboard/api-keys")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._min_interval = min_interval
        self._sleep = sleep
        self._max_retries = max_retries
        self._timeout = timeout
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict | None = None) -> dict:
        url = self._base + path
        headers = {"X-API-Key": self._key, "Accept": "application/json"}
        for attempt in range(self._max_retries + 1):
            self._throttle()
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            if resp.status_code == 429 and attempt < self._max_retries:
                self._sleep(float(resp.headers.get("Retry-After", 5) or 5))
                continue
            try:
                payload = resp.json()
            except ValueError:
                raise DarkerDBError(resp.status_code, "non-JSON response")
            if resp.status_code >= 400:
                raise DarkerDBError(resp.status_code, _error_message(payload))
            return payload
        raise DarkerDBError(429, "rate limited after retries")

    def find_item_id(self, name: str) -> str:
        body = self.get("/items", {"name": name, "limit": 50}).get("body") or []
        rows = body if isinstance(body, list) else [body]
        exact = [r for r in rows if str(r.get("name", "")).lower() == name.lower()]
        chosen = exact or rows
        if not chosen:
            raise DarkerDBError(404, f"no item named {name!r} in /v2/items")
        return str(chosen[0]["id"])

    def iter_pages(self, path: str, params: dict, max_pages: int | None = None) -> Iterator[list[dict]]:
        """Yield the `body` of each page, following pagination.next until it is null or max_pages is hit."""
        params = dict(params)
        pages = 0
        while True:
            payload = self.get(path, params)
            body = payload.get("body") or []
            yield list(body) if isinstance(body, list) else [body]
            pages += 1
            nxt = (payload.get("pagination") or {}).get("next")
            if not nxt or (max_pages is not None and pages >= max_pages):
                return
            if str(nxt).startswith("http"):
                # page-based endpoints (e.g. /market) return the full URL of the next page
                from urllib.parse import parse_qsl, urlsplit
                params = dict(parse_qsl(urlsplit(nxt).query))
            else:
                params["cursor"] = nxt

    def iter_market(self, item_id: str | None, since: str | None = None, limit: int = MARKET_MAX_LIMIT,
                    sort: str = "created_at:desc", max_pages: int | None = None) -> Iterator[list[dict]]:
        """Yield pages of raw listing rows (all items when item_id is None)."""
        params: dict = {"limit": min(limit, MARKET_MAX_LIMIT), "sort": sort}
        if item_id:
            params["item_id"] = item_id
        if since:
            params["from"] = since
        return self.iter_pages("/market", params, max_pages)
