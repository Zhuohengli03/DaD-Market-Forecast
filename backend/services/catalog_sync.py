"""Pull the DarkerDB item catalog and summarize recent market activity per item."""
from __future__ import annotations

import statistics
from typing import Callable

from src.api.darkerdb_client import MARKET_MAX_LIMIT, DarkerDBError, listing_item_ref, normalize_listing
from ..data.catalog import CatalogStore

ITEMS_PAGE_LIMIT = 200
Logger = Callable[[str], None]


def sync_catalog(client, store: CatalogStore, log: Logger) -> int:
    patch = None
    try:
        patch = (client.get("/patches/current").get("body") or {}).get("id")
        log(f"current game patch: {patch}")
    except DarkerDBError as exc:
        log(f"could not read current patch ({exc}); continuing")
    rows: list[dict] = []
    for page_no, page in enumerate(client.iter_pages("/items", {"limit": ITEMS_PAGE_LIMIT}), start=1):
        rows.extend(page)
        log(f"items page {page_no}: {len(rows)} items so far")
    store.save_items(rows, patch)
    log(f"catalog saved: {len(rows)} items")
    return len(rows)


def scan_market(client, store: CatalogStore, since: str | None, max_pages: int, log: Logger) -> int:
    acc: dict[str, dict] = {}
    params = {"limit": MARKET_MAX_LIMIT, "sort": "created_at:desc"}
    if since:
        params["from"] = since
    for page_no, page in enumerate(client.iter_pages("/market", params, max_pages), start=1):
        for raw in page:
            iid, name = listing_item_ref(raw)
            if not iid:
                continue
            try:
                row = normalize_listing(raw, name or iid)
            except (KeyError, TypeError, ValueError):
                continue
            a = acc.setdefault(iid, {"name": row["item"], "prices": [], "sold": 0, "last_seen": row["created_at"]})
            a["prices"].append(row["price_per_unit"])
            a["sold"] += int(row["has_sold"])
            a["last_seen"] = max(a["last_seen"], row["created_at"])
        log(f"market page {page_no}/{max_pages}: {len(acc)} distinct items")
    stats = {iid: {"name": a["name"], "listings": len(a["prices"]), "sold": a["sold"],
                   "min_price_per_unit": min(a["prices"]), "median_price_per_unit": statistics.median(a["prices"]),
                   "last_seen": a["last_seen"]} for iid, a in acc.items()}
    store.save_market(stats, since)
    log(f"market scan saved: {len(stats)} items seen")
    return len(stats)
