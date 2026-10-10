"""Pull the DarkerDB item catalog and find out which item types actually trade."""
from __future__ import annotations

from typing import Callable

from ..data.catalog import CatalogStore

ITEMS_PAGE_LIMIT = 200
ZH_LOCALE = "zh-Hans"
SAVE_EVERY = 25
Logger = Callable[[str], None]


def sync_catalog(client, store: CatalogStore, log: Logger, progress=None) -> int:
    step = progress or (lambda *a, **k: None)
    patch = None
    try:
        body = client.get("/patches/current").get("body")
        if isinstance(body, list):
            body = body[0] if body else None
        patch = (body.get("title") or body.get("id") or body.get("version_raw") or body.get("slug")) \
            if isinstance(body, dict) else (str(body) if body else None)
        log(f"当前游戏版本：{patch}")
    except Exception as exc:  # the patch label is informational; never fail the sync for it
        log(f"读取游戏版本失败（{type(exc).__name__}），继续同步")
    rows: list[dict] = []
    for page_no, page in enumerate(client.iter_pages("/items", {"limit": ITEMS_PAGE_LIMIT}), start=1):
        rows.extend(page)
        log(f"道具第 {page_no} 页：已获取 {len(rows)} 个")
        step("items", len(rows))
    step("names")
    zh = chinese_names(client, log)
    for r in rows:
        if r.get("id") in zh:
            r["name_zh"], r["flavor_zh"] = zh[r["id"]]
    store.save_items(rows, patch)
    log(f"道具库已保存：{len(rows)} 个道具，{len(store.types())} 种类型")
    step("done", items=len(rows), types=len(store.types()))
    return len(rows)


def chinese_names(client, log: Logger) -> dict[str, tuple[str, str | None]]:
    """Simplified Chinese names (locale must be exactly "zh-Hans"); English stays canonical for lookups."""
    out: dict[str, tuple[str, str | None]] = {}
    try:
        for page in client.iter_pages("/items", {"limit": ITEMS_PAGE_LIMIT, "locale": ZH_LOCALE}):
            for r in page:
                if r.get("id") and r.get("name"):
                    out[r["id"]] = (r["name"], r.get("flavor"))
        log(f"已获取 {len(out)} 个道具的中文名称")
    except Exception as exc:
        log(f"获取中文名称失败（{type(exc).__name__}），先显示英文名称")
    return out


def check_tradability(client, store: CatalogStore, log: Logger, force: bool = False, progress=None,
                      season_start: str | None = None) -> int:
    """For every type: total listings ever and the first one (oldest-first query), plus listings this season.

    DarkerDB's is_tradable flag is true for every item, so "ever listed" is the tradability signal. Older
    DarkerDB data mixes all seasons, so the first-listing date and a separate current-season count tell the
    user what the totals contain. Progress is saved as it goes; a new season only re-counts the season column.
    """
    from src.api.darkerdb_client import canonical_ts
    step = progress or (lambda *a, **k: None)
    info = {} if force else {k: dict(v) for k, v in store.read()["trade_info"].items()}
    todo = []
    for arch in store.types():
        cur = info.get(arch)
        need_history = cur is None
        need_season = bool(season_start) and (cur is None or cur.get("season_start") != season_start)
        if need_history or need_season:
            todo.append((arch, need_history, need_season))
    log(f"需要检查 {len(todo)} 种道具（已检查 {len(info)} 种）")
    step("check", 0, len(todo))
    for i, (arch, need_history, need_season) in enumerate(todo, start=1):
        cur = info.setdefault(arch, {"total": 0, "first": None, "season": None, "season_start": None})
        if need_history:
            payload = client.get("/market", {"archetype": arch, "limit": 1, "sort": "created_at:asc"})
            cur["total"] = int((payload.get("pagination") or {}).get("total") or 0)
            body = payload.get("body") or []
            cur["first"] = canonical_ts(body[0]["created_at"]) if body and body[0].get("created_at") else None
        if need_season:
            if cur["total"] == 0:
                cur["season"] = 0
            else:
                payload = client.get("/market", {"archetype": arch, "limit": 1, "from": season_start})
                cur["season"] = int((payload.get("pagination") or {}).get("total") or 0)
            cur["season_start"] = season_start
        step("check", i, len(todo))
        if i % SAVE_EVERY == 0:
            store.save_trade_info(info, complete=False)
            log(f"已检查 {i}/{len(todo)}")
    store.save_trade_info(info, complete=True)
    tradable = sum(1 for v in info.values() if v["total"] > 0)
    log(f"完成：{len(info)} 种道具中有 {tradable} 种在交易行出现过")
    step("done", checked=len(info), tradable=tradable)
    return tradable
