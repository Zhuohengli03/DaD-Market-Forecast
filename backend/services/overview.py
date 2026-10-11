"""Dashboard overview: tracked items grouped by type (all qualities of a ruby together), with prices ready to draw."""
from __future__ import annotations

import math
import re

import pandas as pd

from src.forecast.aggregate import hourly_series
from ..data.catalog import RARITY_ZH, shared_name

SPARK_HOURS = 168               # a week on the card
SPARK_STEP = 4                  # one point per 4 hours: 42 points
UPDATE_DAYS = 14                # an update naming the item this recently shows on its card
RARITY_ORDER = ["poor", "common", "uncommon", "rare", "epic", "legendary", "unique", "artifact"]
_PAREN = re.compile(r"^[\s（(]*(.*?)[)）\s]*$")


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def summarize(hourly: pd.DataFrame) -> dict:
    """Latest hourly median, change vs the last price at least 24 h earlier, and the last 48 hours for a sparkline."""
    prices = hourly["median_price"].astype(float) if len(hourly) else pd.Series(dtype=float)
    valid = prices.dropna()
    n = SPARK_HOURS // SPARK_STEP
    week = prices.iloc[-SPARK_HOURS:]
    spark = [_num(v) for v in week.resample(f"{SPARK_STEP}h", origin="end").median().iloc[-n:]] if len(week) else []
    spark = [None] * (n - len(spark)) + spark
    if valid.empty:
        return {"latest": None, "change24": None, "spark": spark, "latest_ts": None, "latest_listings": None}
    last_ts, latest = valid.index[-1], float(valid.iloc[-1])
    before = valid[valid.index <= last_ts - pd.Timedelta(hours=24)]
    change = None
    if not before.empty and before.iloc[-1]:
        change = round(latest / float(before.iloc[-1]) - 1, 6)
    # which hour the latest price is from, and how many listings it is the median of (shown beside the number)
    count = int(hourly["listings"].loc[last_ts]) if "listings" in hourly.columns else None
    return {"latest": latest, "change24": change, "spark": spark, "latest_ts": last_ts.isoformat(), "latest_listings": count}


def _short_label(variant_name: str | None, base: str | None, rarity: str | None, zh: bool) -> str:
    """"红宝石（裂开）" in group "红宝石" -> "裂开"; potions share one name, so their rarity is the label."""
    if variant_name and base and variant_name != base and variant_name.startswith(base):
        return _PAREN.sub(r"\1", variant_name[len(base):]) or variant_name
    if zh:
        return RARITY_ZH.get(rarity or "", rarity or variant_name or "")
    return (rarity or variant_name or "").title()


def _next_hour(df: pd.DataFrame) -> dict | None:
    """Where next hour's sales should land (P25-P75), or None when there are too few recent sales."""
    from src.forecast.next_hour import next_hour_range
    from src.forecast.pipeline import InsufficientDataError
    try:
        r = next_hour_range(df, check=False)
    except InsufficientDataError:
        return None
    return {"p25": r["prices"]["p25"], "p75": r["prices"]["p75"], "stale_hours": r["stale_hours"]}


def _recent_update(group: dict, patches) -> dict | None:
    """The newest patch naming this item type within UPDATE_DAYS of its newest listing (or of now)."""
    from .announcements import cached_mentions
    found = cached_mentions(group["name"], patches)
    if not found:
        return None
    newest = max((v["last_ts"] for v in group["variants"] if v["last_ts"]), default=None)
    end = pd.Timestamp(newest) if newest else pd.Timestamp.now(tz="UTC")
    end = end if end.tzinfo else end.tz_localize("UTC")
    recent = [m for m in found if end - pd.Timedelta(days=UPDATE_DAYS) <= pd.Timestamp(m["date"]) <= end + pd.Timedelta(days=1)]
    return {"title": recent[0]["title"], "date": recent[0]["date"]} if recent else None


def _trading_windows(st) -> list[tuple[dict, pd.Timestamp, pd.Timestamp | None]]:
    """Every season whose market has opened, newest first: (season, market start, end)."""
    from src.season_calendar import market_start
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for s in st.seasons.list():
        if not s.get("start"):
            continue
        start = pd.Timestamp(market_start(s))
        start = start if start.tzinfo else start.tz_localize("UTC")
        end = pd.Timestamp(s["end"]) if s.get("end") else None
        if start <= now:
            out.append((s, start, end if end is None or end.tzinfo else end.tz_localize("UTC")))
    return sorted(out, key=lambda w: w[1], reverse=True)


def build_overview(st, season_id: str | None = None, all_seasons: bool = False) -> dict:
    """Cards for the chosen season only (the current one when none is given); an item with no listings in that
    season shows no price, however much it has from earlier seasons. KeyError when the season is unknown.

    `all_seasons`: every item at its own newest season with trading data (records from before a market opened
    never count), each variant labelled with that season."""
    from ..data.repository import ListingQuery
    from .collection import pick_season
    from src.season_calendar import market_start
    stats = {i.name: i for i in st.item_stats.get()}
    chosen = None if all_seasons else pick_season(st, season_id)
    windows = _trading_windows(st) if all_seasons else []
    season = pd.Timestamp(market_start(chosen)) if chosen else None
    season_end = pd.Timestamp(chosen["end"]) if chosen and chosen.get("end") else None
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
        summary = {"latest": None, "change24": None, "spark": [None] * (SPARK_HOURS // SPARK_STEP), "next_hour": None,
                   "latest_ts": None, "latest_listings": None}
        season_rows = 0
        shown = chosen
        if s and s.rows and s.last_ts and all_seasons:
            season_rows, last = 0, None
            for w, w_start, w_end in windows:          # newest season first; the first holding listings wins
                newest, n = st.repo.fetch(it["name"], ListingQuery(start=w_start, end=w_end, page_size=1))
                if n:
                    shown, season_rows, last, season = w, n, pd.Timestamp(newest[0]["created_at"]), w_start
                    break
            if season_rows:
                last = last if last.tzinfo else last.tz_localize("UTC")
                df = st.repo.fetch_all(it["name"], max(last - pd.Timedelta(hours=SPARK_HOURS + 24), season), last + pd.Timedelta(seconds=1))
                summary = {**summarize(hourly_series(df)), "next_hour": _next_hour(df)}
            else:
                shown = None
        elif s and s.rows and s.last_ts:
            if season is None:
                season_rows, last = s.rows, pd.Timestamp(s.last_ts)
            else:
                # stored totals can include other seasons (older collectors walked back 200 pages), so count this one;
                # the newest row inside the window is the card's "now" (an ended season stops at its end)
                newest, season_rows = st.repo.fetch(it["name"], ListingQuery(start=season, end=season_end, page_size=1))
                last = pd.Timestamp(newest[0]["created_at"]) if newest else None
            if season_rows:
                last = last if last.tzinfo else last.tz_localize("UTC")
                start = last - pd.Timedelta(hours=SPARK_HOURS + 24)
                if season is not None:
                    start = max(start, season)
                df = st.repo.fetch_all(it["name"], start, last + pd.Timedelta(seconds=1))
                summary = {**summarize(hourly_series(df)), "next_hour": _next_hour(df)}
        g["icon_url"] = g["icon_url"] or icon
        g["variants"].append({"name": it["name"], "label": label, "label_en": label_en, "rarity": rarity,
                              "rows": s.rows if s else 0, "season_rows": season_rows, "last_ts": s.last_ts if s else None,
                              "season_id": shown["id"] if shown and season_rows else None,
                              "season_name": shown["name"] if shown and season_rows else None, **summary})
    patches = st.announcements
    for g in groups.values():
        g["recent_update"] = _recent_update(g, patches)
        g["variants"].sort(key=lambda v: RARITY_ORDER.index(v["rarity"]) if v["rarity"] in RARITY_ORDER else 99)
        g["rows"] = sum(v["rows"] for v in g["variants"])
        g["season_rows"] = sum(v["season_rows"] for v in g["variants"])
    season_out = {"id": chosen["id"], "name": chosen["name"], "start": chosen["start"], "end": chosen.get("end")} if chosen else None
    return {"season": season_out, "groups": list(groups.values())}
