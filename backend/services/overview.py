"""Dashboard overview: tracked items grouped by type (all qualities of a ruby together), with prices ready to draw."""
from __future__ import annotations

import math
import re

import pandas as pd

from src.forecast.aggregate import hourly_series
from ..data.catalog import RARITY_ZH, shared_name

SPARK_HOURS = 48
RARITY_ORDER = ["poor", "common", "uncommon", "rare", "epic", "legendary", "unique", "artifact"]
_PAREN = re.compile(r"^[\s（(]*(.*?)[)）\s]*$")


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def summarize(hourly: pd.DataFrame) -> dict:
    """Latest hourly median, change vs the last price at least 24 h earlier, and the last 48 hours for a sparkline."""
    prices = hourly["median_price"].astype(float) if len(hourly) else pd.Series(dtype=float)
    valid = prices.dropna()
    spark = [_num(v) for v in prices.iloc[-SPARK_HOURS:]]
    spark = [None] * (SPARK_HOURS - len(spark)) + spark
    if valid.empty:
        return {"latest": None, "change24": None, "spark": spark}
    last_ts, latest = valid.index[-1], float(valid.iloc[-1])
    before = valid[valid.index <= last_ts - pd.Timedelta(hours=24)]
    change = None
    if not before.empty and before.iloc[-1]:
        change = round(latest / float(before.iloc[-1]) - 1, 6)
    return {"latest": latest, "change24": change, "spark": spark}


def _short_label(variant_name: str | None, base: str | None, rarity: str | None, zh: bool) -> str:
    """"红宝石（裂开）" in group "红宝石" -> "裂开"; potions share one name, so their rarity is the label."""
    if variant_name and base and variant_name != base and variant_name.startswith(base):
        return _PAREN.sub(r"\1", variant_name[len(base):]) or variant_name
    if zh:
        return RARITY_ZH.get(rarity or "", rarity or variant_name or "")
    return (rarity or variant_name or "").title()


def build_overview(st) -> dict:
    from ..data.repository import ListingQuery
    from .cost import latest_season_start
    stats = {i.name: i for i in st.item_stats.get()}
    season = latest_season_start(st)
    types = st.catalog.types()
    groups: dict[str, dict] = {}
    for it in st.items.list():
        found = st.catalog.variant(it["item_id"]) if it.get("item_id") else None
        if found:
            raw, variants = found
            key = raw.get("archetype") or it["name"]
            group_variants = types.get(key, variants)
            name = shared_name([v.get("name") or key for v in group_variants])
            zhs = [v.get("name_zh") for v in group_variants if v.get("name_zh")]
            name_zh = shared_name(zhs) if zhs else None
            g = groups.setdefault(key, {"key": key, "archetype": key, "name": name, "name_zh": name_zh,
                                        "category": raw.get("misc_type") or raw.get("utility_type") or raw.get("item_type"),
                                        "icon_url": None, "variants": []})
            label = _short_label(raw.get("name_zh"), name_zh, raw.get("rarity"), True)
            label_en = _short_label(raw.get("name"), name, raw.get("rarity"), False)
            rarity, icon = raw.get("rarity"), raw.get("icon_url")
        else:
            key = it["name"]
            g = groups.setdefault(key, {"key": key, "archetype": None, "name": it["name"], "name_zh": None,
                                        "category": it.get("category") or "other", "icon_url": None, "variants": []})
            label = label_en = it["name"]
            rarity, icon = None, None
        s = stats.get(it["name"])
        summary = {"latest": None, "change24": None, "spark": [None] * SPARK_HOURS}
        season_rows = 0
        if s and s.rows and s.last_ts:
            # stored totals can include earlier seasons (older collectors walked back 200 pages), so count this one
            season_rows = st.repo.fetch(it["name"], ListingQuery(start=season, page_size=1))[1] if season is not None else s.rows
            last = pd.Timestamp(s.last_ts)
            last = last if last.tzinfo else last.tz_localize("UTC")
            df = st.repo.fetch_all(it["name"], last - pd.Timedelta(hours=SPARK_HOURS + 24), last + pd.Timedelta(seconds=1))
            summary = summarize(hourly_series(df))
        g["icon_url"] = g["icon_url"] or icon
        g["variants"].append({"name": it["name"], "label": label, "label_en": label_en, "rarity": rarity,
                              "rows": s.rows if s else 0, "season_rows": season_rows, "last_ts": s.last_ts if s else None, **summary})
    for g in groups.values():
        g["variants"].sort(key=lambda v: RARITY_ORDER.index(v["rarity"]) if v["rarity"] in RARITY_ORDER else 99)
        g["rows"] = sum(v["rows"] for v in g["variants"])
        g["season_rows"] = sum(v["season_rows"] for v in g["variants"])
    return {"groups": list(groups.values())}
