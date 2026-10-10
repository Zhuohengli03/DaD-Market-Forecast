"""Pull the DarkerDB item catalog and find out which item types actually trade."""
from __future__ import annotations

from typing import Callable

from ..data.catalog import CatalogStore

ITEMS_PAGE_LIMIT = 200
SAVE_EVERY = 25
Logger = Callable[[str], None]


def sync_catalog(client, store: CatalogStore, log: Logger) -> int:
    patch = None
    try:
        body = client.get("/patches/current").get("body")
        if isinstance(body, list):
            body = body[0] if body else None
        patch = (body.get("title") or body.get("id") or body.get("version_raw") or body.get("slug")) \
            if isinstance(body, dict) else (str(body) if body else None)
        log(f"current game patch: {patch}")
    except Exception as exc:  # the patch label is informational; never fail the sync for it
        log(f"could not read current patch ({type(exc).__name__}); continuing")
    rows: list[dict] = []
    for page_no, page in enumerate(client.iter_pages("/items", {"limit": ITEMS_PAGE_LIMIT}), start=1):
        rows.extend(page)
        log(f"items page {page_no}: {len(rows)} items so far")
    store.save_items(rows, patch)
    log(f"catalog saved: {len(rows)} items in {len(store.types())} types")
    return len(rows)


def check_tradability(client, store: CatalogStore, log: Logger, force: bool = False) -> int:
    """Ask the market how many listings each type has ever had. Zero means it cannot be traded.

    DarkerDB's is_tradable flag is true for every item, so this is the reliable signal. One request per
    type; progress is saved as it goes, so an interrupted check resumes where it stopped.
    """
    totals = {} if force else dict(store.read()["trade_totals"])
    todo = [k for k in store.types() if k not in totals]
    log(f"{len(todo)} item types to check ({len(totals)} already done)")
    for i, arch in enumerate(todo, start=1):
        payload = client.get("/market", {"archetype": arch, "limit": 1})
        totals[arch] = int((payload.get("pagination") or {}).get("total") or 0)
        if i % SAVE_EVERY == 0:
            store.save_trade_totals(totals, complete=False)
            log(f"checked {i}/{len(todo)}")
    store.save_trade_totals(totals, complete=True)
    tradable = sum(1 for v in totals.values() if v > 0)
    log(f"done: {tradable} of {len(totals)} types have market listings")
    return tradable
