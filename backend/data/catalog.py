"""Local cache of the DarkerDB item catalog (data/catalog.json), presented as item *types*.

DarkerDB returns one row per rarity variant (2,400+). A type is every variant sharing an `archetype`
(one weapon name, one ore). Types with random stats (any variant has secondary attributes) are equipment;
the rest can be tracked, collected and forecast.
"""
from __future__ import annotations

import re

import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from ..services.envfile import atomic_write

RARITY_ORDER = ["poor", "common", "uncommon", "rare", "epic", "legendary", "unique", "artifact"]
VARIANT_FIELDS = ("id", "rarity", "gear_score", "vendor_price", "num_primary_attributes", "num_secondary_attributes",
                  "icon_url")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def type_key(raw: dict) -> str:
    return str(raw.get("archetype") or raw.get("id"))


def has_stats(variants: list[dict]) -> bool:
    return any((v.get("num_secondary_attributes") or 0) > 0 for v in variants)


_GRADE = re.compile(r"\s*[（(][^（）()]*[)）]\s*$")
RARITY_ZH = {"poor": "粗劣", "common": "普通", "uncommon": "优秀", "rare": "稀有", "epic": "史诗",
             "legendary": "传说", "unique": "独特", "artifact": "神器"}


def shared_name(names: list[str]) -> str:
    """Name for a whole type: gems are "Ruby (Cracked)", "Ruby (Flawed)"…, so the type is "Ruby"."""
    if len(set(names)) <= 1:
        return names[0]
    bases = {_GRADE.sub("", n) for n in names}
    return bases.pop() if len(bases) == 1 else names[0]


def variant_name_zh(raw: dict, variants: list[dict]) -> str | None:
    """Chinese name of one variant; add the rarity when every rarity shares one name (potions, not gems)."""
    zh = raw.get("name_zh")
    if not zh:
        return None
    same = len({v.get("name") for v in variants}) == 1 and len(variants) > 1
    r = RARITY_ZH.get(str(raw.get("rarity") or ""))
    return f"{zh}（{r}）" if same and r else zh

class CatalogStore:
    _DEFAULTS = (("items", []), ("patch", None), ("synced_at", None), ("trade_totals", {}), ("trade_info", {}),
                 ("trade_check_complete", False), ("trade_checked_at", None))

    def __init__(self, path: Path):
        self._path = Path(path)
        self._cache: tuple[int, dict] | None = None

    def read(self) -> dict:
        """Parsed catalog, cached until the file changes (it can be several MB)."""
        if not self._path.exists():
            return {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in self._DEFAULTS}
        mtime = self._path.stat().st_mtime_ns
        if self._cache and self._cache[0] == mtime:
            return self._cache[1]
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for k, v in self._DEFAULTS:
            data.setdefault(k, v)
        self._cache = (mtime, data)
        return data

    def _write(self, data: dict) -> None:
        atomic_write(self._path, json.dumps(data, ensure_ascii=False))

    def count(self) -> int:
        return len(self.read()["items"])

    def save_items(self, rows: list[dict], patch: str | None) -> None:
        data = dict(self.read())
        data.update(items=rows, patch=patch, synced_at=_now())
        self._write(data)

    def save_trade_totals(self, totals: dict[str, int], complete: bool) -> None:
        data = dict(self.read())
        data.update(trade_totals=totals, trade_check_complete=complete, trade_checked_at=_now())
        self._write(data)

    def save_trade_entry(self, arch: str, entry: dict) -> None:
        data = dict(self.read())
        info = {**data["trade_info"], arch: entry}
        totals = data["trade_totals"] if entry.get("total") is None else {**data["trade_totals"], arch: entry["total"]}
        data.update(trade_info=info, trade_totals=totals)
        self._write(data)

    def save_trade_info(self, info: dict[str, dict], complete: bool) -> None:
        """Per type: total listings ever, first listing time, listings in the current season."""
        data = dict(self.read())
        data.update(trade_info=info, trade_totals={k: v["total"] for k, v in info.items()},
                    trade_check_complete=complete, trade_checked_at=_now())
        self._write(data)

    # ----- types -------------------------------------------------------------
    def types(self) -> "OrderedDict[str, list[dict]]":
        groups: OrderedDict[str, list[dict]] = OrderedDict()
        for raw in self.read()["items"]:
            groups.setdefault(type_key(raw), []).append(raw)
        for variants in groups.values():
            variants.sort(key=lambda v: RARITY_ORDER.index(v["rarity"]) if v.get("rarity") in RARITY_ORDER else 99)
        return groups

    def variant(self, item_id: str) -> tuple[dict, list[dict]] | None:
        for variants in self.types().values():
            for v in variants:
                if v.get("id") == item_id:
                    return v, variants
        return None

    def type_variants(self, archetype: str) -> list[dict]:
        return self.types().get(archetype, [])

    def _type_row(self, key: str, variants: list[dict], totals: dict, info: dict | None = None) -> dict:
        first = variants[0]
        names = [v.get("name") or key for v in variants]
        names_zh = [v.get("name_zh") for v in variants if v.get("name_zh")]
        ti = (info or {}).get(key) or {}
        return {
            "archetype": key, "name": shared_name(names), "name_zh": shared_name(names_zh) if names_zh else None, "item_type": first.get("item_type"),
            "category": first.get("misc_type") or first.get("utility_type") or first.get("item_type"),
            "slot_type": first.get("slot_type"), "icon_url": next((v.get("icon_url") for v in reversed(variants) if v.get("icon_url")), None),
            "has_stats": has_stats(variants), "rarities": [v.get("rarity") for v in variants],
            "listings_total": ti.get("total", totals.get(key)),
            "first_listed": ti.get("first"), "season_listings": ti.get("season"), "season_first": ti.get("season_first"),
            "variants": [{**{k: v.get(k) for k in VARIANT_FIELDS}, "raw": v} for v in variants],
        }

    def type_row(self, archetype: str) -> dict | None:
        """One type with all its variants and parameters, for the item detail page."""
        variants = self.types().get(archetype)
        if not variants:
            return None
        d = self.read()
        return self._type_row(archetype, variants, d["trade_totals"], d["trade_info"])

    def visible_types(self) -> list[dict]:
        d = self.read()
        totals, info = d["trade_totals"], d["trade_info"]
        # hide only types we checked and found never listed; unchecked types stay visible
        return [self._type_row(k, v, totals, info) for k, v in self.types().items() if totals.get(k, 1) != 0]

    def query(self, q: str = "", item_type: str = "", has_stats: bool | None = None,
              page: int = 1, page_size: int = 50) -> tuple[list[dict], int, dict]:
        all_rows = self.visible_types()
        rows = []
        for r in all_rows:
            if q and q.lower() not in r["name"].lower() and q.lower() not in r["archetype"].lower() \
                    and q not in (r["name_zh"] or ""):
                continue
            if item_type and (r["item_type"] or "").lower() != item_type.lower():
                continue
            if has_stats is not None and r["has_stats"] != has_stats:
                continue
            rows.append(r)
        # this season first (older DarkerDB data mixes every season), then all-time, then name
        rows.sort(key=lambda r: (r["season_listings"] is None, -(r["season_listings"] or 0),
                                 -(r["listings_total"] or 0), r["name"]))
        facets = {"item_type": sorted({r["item_type"] for r in all_rows if r["item_type"]}),
                  "category": sorted({r["category"] for r in all_rows if r["category"]})}
        start = (page - 1) * page_size
        return rows[start:start + page_size], len(rows), facets

    def meta(self) -> dict:
        d = self.read()
        groups = self.types()
        visible = self.visible_types()
        return {"patch": d["patch"], "synced_at": d["synced_at"], "items": len(d["items"]), "types": len(visible),
                "types_total": len(groups), "trade_checked": sum(1 for k in groups if k in d["trade_info"] or k in d["trade_totals"]),
                "season_start": next((v.get("season_start") for v in d["trade_info"].values()), None),
                "trade_check_complete": d["trade_check_complete"], "trade_checked_at": d["trade_checked_at"],
                "with_stats": sum(1 for r in visible if r["has_stats"])}
