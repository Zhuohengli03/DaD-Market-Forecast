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


def _history(client, arch: str) -> dict:
    """All-time listing count and the first listing, in one request (oldest first, one row)."""
    from src.api.darkerdb_client import canonical_ts
    payload = client.get("/market", {"archetype": arch, "limit": 1, "sort": "created_at:asc"})
    body = payload.get("body") or []
    return {"total": int((payload.get("pagination") or {}).get("total") or 0),
            "first": canonical_ts(body[0]["created_at"]) if body and body[0].get("created_at") else None}


def check_tradability(client, store: CatalogStore, log: Logger, force: bool = False, progress=None) -> int:
    """One request per unchecked type: has it ever been listed, how often, and when first.

    DarkerDB's is_tradable flag is true for every item, so "ever listed" is the tradability signal. Totals from an
    earlier check are reused without a request; the free DarkerDB plan only allows about 10,000 requests a month.
    Current-season counts are fetched per type on demand (type_history).
    """
    step = progress or (lambda *a, **k: None)
    d = store.read()
    info = {} if force else {k: dict(v) for k, v in d["trade_info"].items()}
    if not force:
        for arch, total in d["trade_totals"].items():
            info.setdefault(arch, {"total": int(total), "first": None, "season": None, "season_start": None})
    todo = [a for a in store.types() if a not in info]
    log(f"需要检查 {len(todo)} 种道具（已检查 {len(info)} 种）")
    step("check", 0, len(todo))
    for i, arch in enumerate(todo, start=1):
        info[arch] = {**_history(client, arch), "season": None, "season_start": None}
        step("check", i, len(todo))
        if i % SAVE_EVERY == 0:
            store.save_trade_info(info, complete=False)
            log(f"已检查 {i}/{len(todo)}")
    store.save_trade_info(info, complete=True)
    tradable = sum(1 for v in info.values() if v["total"] > 0)
    log(f"完成：{len(info)} 种道具中有 {tradable} 种在交易行出现过")
    step("done", checked=len(info), tradable=tradable)
    return tradable


def type_history(client, store: CatalogStore, arch: str, season_start: str | None) -> dict:
    """First listing date and current-season count for one type; at most two requests, cached until a new season."""
    if arch not in store.types():
        raise KeyError(arch)
    cur = dict(store.read()["trade_info"].get(arch) or {"total": None, "first": None, "season": None, "season_start": None})
    if cur.get("total") is None or (cur["total"] > 0 and not cur.get("first")):
        cur.update(_history(client, arch))
    if season_start and cur.get("season_start") != season_start:
        if cur["total"] == 0:
            cur["season"] = 0
        else:
            payload = client.get("/market", {"archetype": arch, "limit": 1, "from": season_start})
            cur["season"] = int((payload.get("pagination") or {}).get("total") or 0)
        cur["season_start"] = season_start
    store.save_trade_entry(arch, cur)
    return cur
