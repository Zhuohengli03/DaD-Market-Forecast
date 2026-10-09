"""Generic market collector for the DarkerDB v2 API.

    python -m src.api.collect --item "Iron Ore"            # fetch new listings -> Postgres -> CSV
    python -m src.api.collect --item "Iron Ore" --probe    # print the first raw row and exit

The per-item scripts in src/api/*_API.py are thin wrappers around main().
"""
from __future__ import annotations

import argparse
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


def record_key(row: dict) -> str:
    return f"{row['id']}_{row['created_at']}"


def collect_pages(client, item_id: str, item_name: str, since: str | None, seen: set[str],
                  max_pages: int = 200, max_empty_pages: int = 3, log: Logger = print) -> list[dict]:
    """Walk the market cursor, returning normalized rows not already in `seen`.

    Stops early after `max_empty_pages` consecutive pages without new rows (listings are
    sorted newest first, so once we hit known rows the rest is history we already have).
    """
    new_rows: list[dict] = []
    empty_streak = 0
    for page_no, page in enumerate(client.iter_market(item_id, since=since, max_pages=max_pages), start=1):
        added = 0
        for raw in page:
            try:
                row = normalize_listing(raw, item_name)
            except (KeyError, TypeError, ValueError) as exc:
                log(f"skipping malformed row ({type(exc).__name__}: {exc}): {json.dumps(raw)[:200]}")
                continue
            key = record_key(row)
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(row)
            added += 1
        log(f"page {page_no}: {added} new / {len(page)} rows")
        if added == 0:
            empty_streak += 1
            if empty_streak >= max_empty_pages:
                log(f"{max_empty_pages} pages without new rows, stopping")
                break
        else:
            empty_streak = 0
    return new_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect DarkerDB v2 market listings for one item")
    p.add_argument("--item", required=True, help='display name, e.g. "Iron Ore"')
    p.add_argument("--item-id", default=None, help="skip the /items lookup, e.g. id.item.iron_ore_1001")
    p.add_argument("--since", default=None, help="only listings created at/after this timestamp (ISO or YYYY-MM-DD)")
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--probe", action="store_true", help="print the first raw market row and exit")
    p.add_argument("--no-csv", action="store_true", help="skip the CSV export after inserting")
    return p.parse_args(argv)


def _load_existing(db) -> set[str]:
    cursor = db.connector.cursor()
    cursor.execute(f"SELECT item_id, created_at FROM {db.items}")
    keys = {f"{r[0]}_{r[1]}" for r in cursor.fetchall()}
    cursor.close()
    return keys


def export_csv(db, table: str, csv_path: Path, log: Logger = print) -> bool:
    import pandas as pd
    cursor = db.connector.cursor()
    cursor.execute(f"SELECT item_id, item, quantity, price_per_unit, price, has_sold, created_at, insert_time "
                   f"FROM {table} ORDER BY created_at DESC")
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    if not rows:
        log("no rows in the database, CSV not written")
        return False
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8")
    log(f"exported {len(rows)} rows to {csv_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src.database.Database_connect import DarkerMarketDB
    from src.database.config import config

    log = print
    try:
        client = DarkerDBClient(config.DARKER_MARKET_API_KEY)
        item_id = args.item_id or client.find_item_id(args.item)
        log(f"item {args.item!r} -> {item_id}")
        if args.probe:
            first = next(client.iter_market(item_id, since=args.since, limit=1, max_pages=1), [])
            print(json.dumps(first[:1], indent=2, ensure_ascii=False))
            if first:
                print("normalized:", json.dumps(normalize_listing(first[0], args.item), ensure_ascii=False))
            return 0
    except DarkerDBError as exc:
        log(f"DarkerDB API error: {exc}")
        if exc.code == 401:
            log("Create a v2 key at https://darkerdb.com/dashboard/api-keys (scopes darkerdb.live + darkerdb.data) "
                "and put it in .env as DARKER_MARKET_API_KEY")
        return 2

    table = args.item.replace(" ", "_").lower()
    db = DarkerMarketDB(items=table)
    if not db.connect():
        log("database connection failed")
        return 3
    try:
        seen = _load_existing(db)
        log(f"{len(seen)} existing records loaded for de-duplication")
        try:
            rows = collect_pages(client, item_id, args.item, args.since, seen, max_pages=args.max_pages, log=log)
        except DarkerDBError as exc:
            log(f"DarkerDB API error: {exc}")
            return 2
        if rows:
            db.add_data(rows)
            inserted = db.batch_insert_all_data()
            log(f"inserted {inserted} new rows into {table}")
        else:
            log("no new listings")
        if not args.no_csv:
            export_csv(db, table, PROJECT_ROOT / f"{table}.csv", log)
    finally:
        db.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
