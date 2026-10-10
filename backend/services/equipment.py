"""Live equipment market queries: listings of one item type filtered by rarity and stat ranges."""
from __future__ import annotations

import statistics
import time
from collections import defaultdict

from src.api.darkerdb_client import MARKET_MAX_LIMIT, canonical_ts, listing_item_ref

CACHE_SECONDS = 300
_attr_cache: dict[str, tuple[float, dict]] = {}
_query_cache: dict[str, tuple[float, dict]] = {}


def _range(lo, hi) -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}:{hi:g}"
    if lo is not None:
        return f">={lo:g}"
    return f"<={hi:g}"


def build_market_params(archetype: str, rarity: str | None, filters: list[dict], sold_only: bool,
                        since: str | None, sort: str = "created_at:desc") -> dict:
    p = {"archetype": archetype, "limit": MARKET_MAX_LIMIT, "sort": sort}
    if rarity:
        p["rarity"] = rarity
    if sold_only:
        p["has_sold"] = "true"
    if since:
        p["from"] = since
    for f in filters:
        if f.get("min") is None and f.get("max") is None:
            continue
        p[f"{f['kind']}[{f['name']}]"] = _range(f.get("min"), f.get("max"))
    return {"archetype": p.pop("archetype"), **({"rarity": p.pop("rarity")} if "rarity" in p else {}),
            **({"has_sold": p.pop("has_sold")} if "has_sold" in p else {}), **p}


def normalize_equipment_row(raw: dict) -> dict:
    _, name = listing_item_ref(raw)
    qty = int(raw.get("quantity") or 1)
    price = float(raw["price"])
    ppu = raw.get("price_per_unit")
    return {"id": str(raw["id"]), "item_id": raw.get("item_id"), "name": name, "rarity": raw.get("rarity"),
            "quantity": qty, "price": price, "price_per_unit": float(ppu) if ppu is not None else price / qty,
            "has_sold": bool(raw.get("has_sold")), "listing_state": raw.get("listing_state"),
            "created_at": canonical_ts(raw["created_at"]), "attributes": raw.get("attributes") or {}}


def daily_stats(rows: list[dict]) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["created_at"][:10]].append(r)
    out = []
    for day in sorted(by_day):
        prices = [r["price_per_unit"] for r in by_day[day]]
        sold = [r["price_per_unit"] for r in by_day[day] if r["has_sold"]]
        out.append({"day": day, "listings": len(prices), "sold": len(sold), "median": statistics.median(prices),
                    "low": min(prices), "high": max(prices), "sold_median": statistics.median(sold) if sold else None})
    return out


def attribute_labels(client) -> dict:
    """id-less attribute name -> {label, percent}. Cached for the process lifetime of CACHE_SECONDS."""
    hit = _attr_cache.get("all")
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    labels: dict[str, dict] = {}
    for page in client.iter_pages("/attributes", {"limit": 250}):
        for a in page:
            key = str(a.get("id", "")).replace("id.attribute.", "")
            labels[key] = {"label": a.get("name") or key, "label_zh": None, "percent": bool(a.get("is_percentage"))}
    try:
        for page in client.iter_pages("/attributes", {"limit": 250, "locale": "zh-Hans"}):
            for a in page:
                key = str(a.get("id", "")).replace("id.attribute.", "")
                if key in labels and a.get("name"):
                    labels[key]["label_zh"] = a["name"]
    except Exception:
        pass  # Chinese labels are a nicety; English still works
    _attr_cache["all"] = (time.monotonic(), labels)
    return labels


def stats_for_type(client, archetype: str, sample_pages: int = 2) -> list[dict]:
    """Stats that appear on this type's listings, with observed value ranges, from a small recent sample."""
    seen: dict[str, list[float]] = defaultdict(list)
    for page in client.iter_pages("/market", {"archetype": archetype, "limit": MARKET_MAX_LIMIT,
                                              "sort": "created_at:desc"}, max_pages=sample_pages):
        for raw in page:
            for k, v in (raw.get("attributes") or {}).items():
                if isinstance(v, (int, float)):
                    seen[k].append(float(v))
    labels = attribute_labels(client)
    out = []
    for key, vals in seen.items():
        kind, _, name = key.partition("_")
        if kind not in ("primary", "secondary"):
            continue
        meta = labels.get(name, {"label": name.replace("_", " ").title(), "label_zh": None, "percent": False})
        out.append({"key": key, "kind": kind, "name": name, "label": meta["label"], "label_zh": meta.get("label_zh"),
                    "percent": meta["percent"],
                    "min": min(vals), "max": max(vals), "seen": len(vals)})
    out.sort(key=lambda s: (s["kind"] != "secondary", -s["seen"], s["label"]))
    return out


def query_market(client, params: dict, max_pages: int) -> dict:
    key = repr(sorted(params.items())) + f"|{max_pages}"
    hit = _query_cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    rows: list[dict] = []
    for page in client.iter_pages("/market", params, max_pages=max_pages):
        for raw in page:
            try:
                rows.append(normalize_equipment_row(raw))
            except (KeyError, TypeError, ValueError):
                continue
    result = {"listings": rows, "daily": daily_stats(rows), "count": len(rows),
              "covered_from": min((r["created_at"] for r in rows), default=None),
              "covered_to": max((r["created_at"] for r in rows), default=None),
              "truncated": len(rows) >= max_pages * MARKET_MAX_LIMIT}
    _query_cache[key] = (time.monotonic(), result)
    return result
