"""How many DarkerDB requests (credits) a task will use, always rounded up.

Collection pages hold 50 listings, so the cost of collecting an item depends on how many new listings appeared since
the newest one stored. That is estimated from the item's own recent listing rate (the faster of the last day and the
last week) times a safety margin, plus one page for the overlap with stored data. Items without local data fall back
to the current-season count of their whole type (an upper bound for one rarity) or to the page cap.
"""
from __future__ import annotations

import math

import pandas as pd

from src.scheduler.task_config import item_schedule

PAGE = 50
MAX_PAGES = 200               # collector default page budget per run
SAFETY = 1.5                  # listing rates swing; never under-promise
UNKNOWN_PER_RUN = 2           # scheduled run of an item with no history yet (flagged to the user)
CATALOG_ITEMS_GUESS = 3000    # before the first catalog download
MONTH_SECONDS = 30 * 86400
_WINDOWS_H = (24, 168)


def _pages(listings: float) -> int:
    return math.ceil(max(listings, 0) / PAGE) + 1    # +1: the page that overlaps what is already stored


def latest_season_start(st) -> pd.Timestamp | None:
    starts = [s["start"] for s in st.seasons.list() if s.get("start")]
    return pd.Timestamp(max(starts, key=pd.Timestamp)) if starts else None


def _stats(st, name: str):
    return next((i for i in st.item_stats.get() if i.name == name), None)


def _rate_per_hour(st, name: str, last: pd.Timestamp) -> float:
    """Listings per hour around the newest stored one: the faster of the last day and the last week."""
    from ..data.repository import ListingQuery
    rates = []
    for h in _WINDOWS_H:
        count = st.repo.fetch(name, ListingQuery(start=last - pd.Timedelta(hours=h), end=last + pd.Timedelta(seconds=1), page_size=1))[1]
        rates.append(count / h)
    return max(rates)


def _type_season(st, item_id: str | None) -> int | None:
    """This-season listings of the item's whole type, if the catalog has checked it."""
    if not item_id:
        return None
    found = st.catalog.variant(item_id)
    if not found:
        return None
    info = st.catalog.read()["trade_info"].get(found[0].get("archetype")) or {}
    return info.get("season")


def _to_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts if ts.tzinfo else ts.tz_localize("UTC")


def collect_estimate(st, entry: dict, now=None, max_pages: int = MAX_PAGES, hours: float | None = None,
                     since: str | None = None, until: str | None = None) -> dict:
    """Requests for one collection of a tracked item.

    The window is the season's (see collection.collect_range): from `since` to `until` (or now). `hours` overrides
    it for scheduled runs. The rate is the item's own recent listing rate, rounded up.
    """
    cap = max_pages + 1
    stats = _stats(st, entry["name"])
    if stats and stats.rows and stats.last_ts:
        last = _to_ts(stats.last_ts)
        if hours is None:
            if since is None:
                from .collection import collect_range
                since, until = collect_range(st, entry["name"])
            begin = _to_ts(since) if since else last
            stop = _to_ts(until) if until else _to_ts(now or pd.Timestamp.now(tz="UTC"))
            hours = (stop - begin).total_seconds() / 3600
        new = _rate_per_hour(st, entry["name"], last) * max(hours, 1) * SAFETY
        return {"requests": min(_pages(new), cap), "known": True}
    season = _type_season(st, entry.get("item_id"))
    if season is not None and until is None:
        return {"requests": min(_pages(season), cap), "known": True}
    return {"requests": cap, "known": False}


def items_estimate(st, item_ids: list[str], max_pages: int = MAX_PAGES) -> dict:
    """First collection of the chosen items (setup wizard, "add items")."""
    by_id = {it.get("item_id"): it for it in st.items.list() if it.get("item_id")}
    out, unknown = {}, 0
    for iid in dict.fromkeys(item_ids):
        entry = by_id.get(iid) or {"name": f"\0{iid}", "item_id": iid}
        est = collect_estimate(st, entry, max_pages=max_pages)
        out[iid] = est["requests"]
        unknown += not est["known"]
    return {"requests": sum(out.values()), "unknown": unknown, "items": out}


def collect_all_estimate(st) -> dict:
    items = st.items.list(enabled_only=True)
    ests = [collect_estimate(st, it) for it in items]
    return {"requests": sum(e["requests"] for e in ests), "items": len(items), "unknown": sum(not e["known"] for e in ests),
            "per_item": {it["name"]: e["requests"] for it, e in zip(items, ests)}}


def schedule_estimate(st, default_seconds: int) -> dict:
    """Requests a month for scheduled collection, item by item, honouring per-item schedules."""
    per_item, unknown = {}, 0
    for it in st.items.list(enabled_only=True):
        kind, value = item_schedule(it, default_seconds)
        if kind == "seconds":
            runs, gap_h = math.ceil(MONTH_SECONDS / int(value)), int(value) / 3600
        else:
            runs, gap_h = len(value) * 30, 24 / len(value)
        stats = _stats(st, it["name"])
        if stats and stats.rows:
            per_run = collect_estimate(st, it, hours=gap_h)["requests"]
        else:
            season = _type_season(st, it.get("item_id"))
            start = latest_season_start(st)
            if season is not None and start is not None:
                hours_in = max((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 3600, 1)
                per_run = _pages(season / hours_in * gap_h * SAFETY)
            else:
                per_run, unknown = UNKNOWN_PER_RUN, unknown + 1
        per_item[it["name"]] = {"runs": runs, "per_run": per_run}
    return {"per_month": sum(v["runs"] * v["per_run"] for v in per_item.values()), "per_item": per_item, "unknown": unknown}


def sync_cost(store) -> int:
    """Catalog download: the current patch, then every /items page in English and in Chinese."""
    n = store.count() or CATALOG_ITEMS_GUESS
    return 1 + 2 * math.ceil(n / 200)


def check_cost(store) -> int:
    d = store.read()
    return sum(1 for a in store.types() if a not in d["trade_info"] and a not in d["trade_totals"])
