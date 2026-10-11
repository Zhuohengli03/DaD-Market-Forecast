"""Generic market collector for the DarkerDB v2 API.

    python -m src.api.collect --item "Iron Ore"            # fetch new listings -> Postgres -> CSV
    python -m src.api.collect --item "Iron Ore" --probe    # print the first raw row and exit

The per-item scripts in src/api/*_API.py are thin wrappers around main().
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.darkerdb_client import DarkerDBClient, DarkerDBError, normalize_listing  # noqa: E402

Logger = Callable[[str], None]


def emit(stage: str, **kw) -> None:
    """Machine-readable progress for the web app (enabled with DARKER_PROGRESS=1); invisible on the CLI."""
    if os.getenv("DARKER_PROGRESS"):
        print("@@" + json.dumps({"stage": stage, **{k: v for k, v in kw.items() if v is not None}}, ensure_ascii=False), flush=True)


def record_key(row: dict) -> str:
    return f"{row['id']}_{row['created_at']}"


def collect_pages(client, item_id: str, item_name: str, since: str | None, seen: set[str],
                  max_pages: int = 200, max_empty_pages: int = 3, log: Logger = print,
                  sort: str = "created_at:desc", until: str | None = None, walk: dict | None = None) -> list[dict]:
    """Walk the market cursor, returning normalized rows not already in `seen`.

    Stops early after `max_empty_pages` consecutive pages without new rows (listings are
    sorted newest first, so once we hit known rows the rest is history we already have).
    """
    new_rows: list[dict] = []
    empty_streak = 0
    walk = walk if walk is not None else {}
    walk.update(pages=0, newest=None, stopped_early=False)   # how far this run got, for resuming a long backfill
    window = {"until": until} if until else {}
    for page_no, page in enumerate(client.iter_market(item_id, since=since, max_pages=max_pages, sort=sort, **window), start=1):
        added = 0
        for raw in page:
            try:
                row = normalize_listing(raw, item_name)
            except (KeyError, TypeError, ValueError) as exc:
                log(f"跳过格式异常的挂单（{type(exc).__name__}: {exc}）：{json.dumps(raw)[:200]}")
                continue
            if walk["newest"] is None or row["created_at"] > walk["newest"]:
                walk["newest"] = row["created_at"]
            key = record_key(row)
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(row)
            added += 1
        log(f"第 {page_no} 页：新增 {added} 条 / 共 {len(page)} 条")
        walk["pages"] = page_no
        emit("collect", done=page_no, total=max_pages, new=len(new_rows), credits=getattr(client, "credits_used", None))
        if added == 0:
            empty_streak += 1
            if empty_streak >= max_empty_pages:
                log(f"连续 {max_empty_pages} 页没有新挂单，停止采集")
                walk["stopped_early"] = True
                break
        else:
            empty_streak = 0
    return new_rows


def _ts(value: str) -> datetime:
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def current_season_start(data_dir: Path) -> str | None:
    """Start of the latest season in data/seasons.json (set up in the app), or None."""
    try:
        seasons = json.loads((data_dir / "seasons.json").read_text(encoding="utf-8")).get("seasons") or []
    except (OSError, ValueError):
        return None
    from src.season_calendar import market_start
    starts = [market_start(s) for s in seasons if s.get("start")]
    return max(starts, key=_ts) if starts else None


def later_of(a: str | None, b: str | None) -> str | None:
    if not a or not b:
        return a or b
    return a if _ts(a) >= _ts(b) else b


def save_coverage(data_dir: Path, slug: str, key: str, through: str) -> None:
    """data/coverage/<item>.json: {window key: collected through}. Only moves forward."""
    path = data_dir / "coverage" / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        marks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marks = {}
    if key not in marks or _ts(through) > _ts(marks[key]):
        marks[key] = through
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(marks, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect DarkerDB v2 market listings for one item")
    p.add_argument("--item", required=True, help='display name, e.g. "Iron Ore"')
    p.add_argument("--item-id", default=None, help="skip the /items lookup, e.g. id.item.iron_ore_1001")
    p.add_argument("--since", default=None, help="only listings created at/after this timestamp (ISO or YYYY-MM-DD)")
    p.add_argument("--until", default=None, help="only listings created before this timestamp (to collect one past season)")
    p.add_argument("--coverage-key", default=None, help="remember how far this window was collected under this key (a season id)")
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--probe", action="store_true", help="print the first raw market row and exit")
    p.add_argument("--data-dir", default=os.getenv("DARKER_DATA_DIR", str(PROJECT_ROOT / "data")))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.api.storage import open_storage

    log = print
    key = os.getenv("DARKER_MARKET_API_KEY", "")
    try:
        client = DarkerDBClient(key)
        item_id = args.item_id or client.find_item_id(args.item)
        log(f"道具 {args.item} 对应 DarkerDB 编号 {item_id}")
        if args.probe:
            first = next(client.iter_market(item_id, since=args.since, limit=1, max_pages=1), [])
            print(json.dumps(first[:1], indent=2, ensure_ascii=False))
            if first:
                print("normalized:", json.dumps(normalize_listing(first[0], args.item), ensure_ascii=False))
            return 0
    except DarkerDBError as exc:
        log(f"DarkerDB 接口出错：{exc}")
        emit("error", code="auth" if exc.code in (401, 403) else "api", http=exc.code,
             credits=getattr(locals().get("client"), "credits_used", None))
        if exc.code == 401:
            log("DarkerDB API key 缺失或无效。请打开网页应用，在「设置」里填写；也可以到 "
                "https://darkerdb.com/dashboard/api-keys 创建（勾选 darkerdb.live 和 darkerdb.data）。")
        return 2

    from src.api.storage import item_slug
    storage = open_storage(item_slug(args.item), Path(args.data_dir), log)
    try:
        seen = storage.existing_keys()
        # incremental: only ask for listings newer than what is stored. v2 listing ids differ from v1 ids,
        # so re-fetching an already stored period would duplicate it.
        # never walk back through an older season: start at the newest stored listing or the current season's
        # start, whichever is later; an explicit --since (one chosen season) wins
        since = args.since or later_of(storage.latest_created_at(), current_season_start(Path(args.data_dir)))
        log(f"已存有 {len(seen)} 条挂单，从 {since or '最新'} 开始采集")
        try:
            # resuming: walk oldest-first from the newest stored listing, so stopping at the page budget leaves
            # no gap (the next run continues). First run: take the newest pages.
            sort = "created_at:asc" if since else "created_at:desc"
            walk: dict = {}
            rows = collect_pages(client, item_id, args.item, since, seen, max_pages=args.max_pages, log=log, sort=sort,
                                 until=args.until, walk=walk)
        except DarkerDBError as exc:
            log(f"DarkerDB 接口出错：{exc}")
            emit("error", code="auth" if exc.code in (401, 403) else "api", http=exc.code, credits=getattr(client, "credits_used", None))
            return 2
        inserted = storage.insert(rows)
        if args.coverage_key:
            # pages ran out before the budget: the whole window is in; otherwise resume after the newest listing seen
            finished = walk["pages"] < args.max_pages and not walk["stopped_early"]
            through = args.until if finished and args.until else walk["newest"]
            if through:
                save_coverage(Path(args.data_dir), item_slug(args.item), args.coverage_key, through)
        emit("done", new=inserted, credits=getattr(client, "credits_used", None))
        log(f"完成：保存了 {inserted} 条新挂单" if inserted else "完成：没有新挂单")
    finally:
        storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
