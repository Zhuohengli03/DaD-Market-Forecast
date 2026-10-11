"""Read side of the built-in SQLite database, plus a one-time import of legacy CSV exports."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.api.sqlite_db import connect
from src.api.storage import item_slug
from .repository import ItemInfo, ListingQuery, validate_item

_COLS = "item_id, item_name AS item, quantity, price_per_unit, price, has_sold, created_at"


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_out(value: str | None) -> str | None:
    if not value:
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat() if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC").isoformat()


class SqliteListingRepository:
    def __init__(self, db_path: Path, items_config_path: Path):
        self._db = Path(db_path)
        self._config_path = Path(items_config_path)

    def source_name(self) -> str:
        return "sqlite"

    def _names(self) -> list[str]:
        if not self._config_path.exists():
            return []
        cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
        return [it["name"] for it in cfg.get("items", []) if it.get("enabled", True)]

    def list_items(self) -> list[ItemInfo]:
        con = connect(self._db)
        stats = {row[0]: row[1:] for row in con.execute(
            "SELECT item, COUNT(*), MIN(created_at), MAX(created_at) FROM listings GROUP BY item")}
        con.close()
        out = []
        for name in self._names():
            n, lo, hi = stats.get(item_slug(name), (0, None, None))
            out.append(ItemInfo(name, int(n), _utc_out(lo), _utc_out(hi)))
        return out

    @staticmethod
    def _where(item: str, start, end, sold=None, min_qty=None, max_qty=None) -> tuple[str, list]:
        clauses, params = ["item = ?"], [item]
        if start is not None:
            clauses.append("created_at >= ?"); params.append(_iso(start))
        if end is not None:
            clauses.append("created_at < ?"); params.append(_iso(end))
        if sold is not None:
            clauses.append("has_sold = ?"); params.append(int(sold))
        if min_qty is not None:
            clauses.append("quantity >= ?"); params.append(min_qty)
        if max_qty is not None:
            clauses.append("quantity <= ?"); params.append(max_qty)
        return " WHERE " + " AND ".join(clauses), params

    def _key(self, item: str) -> str:
        validate_item(item)
        if item not in self._names():
            raise KeyError(f"no data for item {item!r}")
        return item_slug(item)

    def fetch(self, item: str, q: ListingQuery) -> tuple[list[dict], int]:
        where, params = self._where(self._key(item), q.start, q.end, q.sold, q.min_qty, q.max_qty)
        con = connect(self._db)
        total = con.execute(f"SELECT COUNT(*) FROM listings{where}", params).fetchone()[0]
        cur = con.execute(f"SELECT {_COLS} FROM listings{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                          params + [q.page_size, (q.page - 1) * q.page_size])
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        con.close()
        for r in rows:
            r["has_sold"] = bool(r["has_sold"])
            r["created_at"] = _utc_out(r["created_at"])
        return rows, int(total)

    def fetch_all(self, item: str, start, end) -> pd.DataFrame:
        where, params = self._where(self._key(item), start, end)
        con = connect(self._db)
        df = pd.read_sql_query(f"SELECT {_COLS} FROM listings{where} ORDER BY created_at", con, params=params)
        con.close()
        df["has_sold"] = df["has_sold"].astype(bool)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df


    # ---- the local database page: every stored item, read and delete only ----
    def stored_items(self) -> list[dict]:
        tracked = {item_slug(n): n for n in self._names()}
        con = connect(self._db)
        rows = con.execute("SELECT item, MAX(item_name), COUNT(*), MIN(created_at), MAX(created_at) FROM listings GROUP BY item").fetchall()
        con.close()
        out = [{"key": k, "name": tracked.get(k) or label or k, "tracked": k in tracked, "rows": int(n),
                "first": _utc_out(lo), "last": _utc_out(hi)} for k, label, n, lo, hi in rows]
        out += [{"key": k, "name": n, "tracked": True, "rows": 0, "first": None, "last": None}
                for k, n in tracked.items() if k not in {r["key"] for r in out}]
        return sorted(out, key=lambda r: (-r["rows"], r["name"]))

    def browse(self, key: str, start=None, end=None, page: int = 1, page_size: int = 100) -> tuple[list[dict], int]:
        where, params = self._where(key, start, end)
        con = connect(self._db)
        total = con.execute(f"SELECT COUNT(*) FROM listings{where}", params).fetchone()[0]
        cur = con.execute(f"SELECT item_id AS id, quantity, price_per_unit, price, has_sold, created_at FROM listings{where} "
                          "ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [page_size, (page - 1) * page_size])
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        con.close()
        for r in rows:
            r["has_sold"], r["created_at"] = bool(r["has_sold"]), _utc_out(r["created_at"])
        return rows, int(total)

    def delete(self, key: str, start=None, end=None, ids: list[str] | None = None) -> int:
        """Delete one item's rows: the listed ids, else the time range, else (no range) all of them."""
        where, params = self._where(key, start, end)
        if ids:
            where += f" AND item_id IN ({','.join('?' * len(ids))})"
            params += list(ids)
        con = connect(self._db)
        with con:
            n = con.execute(f"DELETE FROM listings{where}", params).rowcount
        con.close()
        return int(n)

def import_csv_once(settings) -> dict[str, int]:
    """Copy legacy CSV exports (data/ or project root) into the built-in database, once per file."""
    from src.api.storage import SqliteStorage
    from src.api.darkerdb_client import canonical_ts
    if not settings.items_config_path.exists():
        return {}
    cfg = json.loads(settings.items_config_path.read_text(encoding="utf-8"))
    con = connect(settings.sqlite_path)
    done = {r[0] for r in con.execute("SELECT source FROM imports")}
    imported: dict[str, int] = {}
    for it in cfg.get("items", []):
        names = {it.get("csv"), f"{item_slug(it['name'])}.csv"} - {None}
        for d in settings.csv_dirs:
            for n in sorted(names):
                path = Path(d) / n
                if not path.exists() or str(path.resolve()) in done:
                    continue
                df = pd.read_csv(path, dtype={"item_id": str})
                rows = [{"id": r.item_id, "item": r.item, "quantity": r.quantity, "price_per_unit": r.price_per_unit,
                         "price": r.price, "has_sold": str(r.has_sold).lower() in ("true", "1", "t"),
                         "created_at": canonical_ts(r.created_at)} for r in df.itertuples()]
                SqliteStorage(settings.sqlite_path, item_slug(it["name"])).insert(rows)
                with con:
                    con.execute("INSERT OR REPLACE INTO imports VALUES (?, ?, ?)",
                                (str(path.resolve()), len(rows), datetime.now(timezone.utc).isoformat()))
                imported[it["name"]] = imported.get(it["name"], 0) + len(rows)
    con.close()
    return imported
