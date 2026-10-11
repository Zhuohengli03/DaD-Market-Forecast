from __future__ import annotations

import json

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql

from ..settings import Settings
from src.api.storage import item_slug
from .repository import ItemInfo, ListingQuery, table_name

_COLS = "item_id, item, quantity, price_per_unit, price, has_sold, created_at"
_TS = "created_at::timestamptz"  # legacy tables store created_at as VARCHAR ISO strings


def _utc_iso(value) -> str | None:
    """ISO string for a VARCHAR, naive or tz-aware timestamp coming back from Postgres."""
    if value is None:
        return None
    t = pd.Timestamp(value)
    t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    return t.isoformat()


class PostgresListingRepository:
    def __init__(self, settings: Settings):
        self._dsn = dict(host=settings.db_host, dbname=settings.db_database, user=settings.db_user,
                         password=settings.db_password, port=settings.db_port,
                         connect_timeout=settings.db_connect_timeout)
        self._items_config_path = settings.items_config_path

    def source_name(self) -> str:
        return "postgres"

    def _conn(self):
        return psycopg2.connect(**self._dsn)

    def probe(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")

    def _configured_items(self) -> list[str]:
        if not self._items_config_path.exists():
            return []
        cfg = json.loads(self._items_config_path.read_text(encoding="utf-8"))
        return [it["name"] for it in cfg.get("items", []) if it.get("enabled", True)]

    def list_items(self) -> list[ItemInfo]:
        out = []
        with self._conn() as c, c.cursor() as cur:
            for name in self._configured_items():
                tbl = table_name(name)
                cur.execute("SELECT to_regclass(%s)", (tbl,))
                if cur.fetchone()[0] is None:
                    out.append(ItemInfo(name, 0, None, None))
                    continue
                cur.execute(sql.SQL(f"SELECT COUNT(*), MIN({_TS}), MAX({_TS}) FROM {{}}").format(sql.Identifier(tbl)))
                n, lo, hi = cur.fetchone()
                out.append(ItemInfo(name, int(n), _utc_iso(lo), _utc_iso(hi)))
        return out

    @staticmethod
    def _where(start, end, sold=None, min_qty=None, max_qty=None) -> tuple[str, list]:
        clauses, params = [], []
        if start is not None:
            clauses.append(f"{_TS} >= %s"); params.append(start.to_pydatetime())
        if end is not None:
            clauses.append(f"{_TS} < %s"); params.append(end.to_pydatetime())
        if sold is not None:
            clauses.append("has_sold = %s"); params.append(sold)
        if min_qty is not None:
            clauses.append("quantity >= %s"); params.append(min_qty)
        if max_qty is not None:
            clauses.append("quantity <= %s"); params.append(max_qty)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def fetch(self, item: str, q: ListingQuery) -> tuple[list[dict], int]:
        tbl = table_name(item)
        where, params = self._where(q.start, q.end, q.sold, q.min_qty, q.max_qty)
        with self._conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT COUNT(*) AS n FROM {}" + where).format(sql.Identifier(tbl)), params)
            total = int(cur.fetchone()["n"])
            cur.execute(sql.SQL(f"SELECT {_COLS} FROM {{}}{where} ORDER BY {_TS} DESC LIMIT %s OFFSET %s").format(sql.Identifier(tbl)),
                        params + [q.page_size, (q.page - 1) * q.page_size])
            rows = []
            for r in cur.fetchall():
                r = dict(r)
                r["item_id"] = str(r["item_id"])
                r["created_at"] = _utc_iso(r["created_at"])
                rows.append(r)
        return rows, total

    def fetch_all(self, item: str, start, end) -> pd.DataFrame:
        tbl = table_name(item)
        where, params = self._where(start, end)
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql.SQL(f"SELECT {_COLS} FROM {{}}{where} ORDER BY {_TS}").format(sql.Identifier(tbl)), params)
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=[d[0] for d in cur.description]) if rows else \
            pd.DataFrame(columns=_COLS.split(", "))
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df


    # ---- the local database page: tracked items' tables, read and delete only ----
    def _name(self, key: str) -> str | None:
        return next((n for n in self._configured_items() if item_slug(n) == key), None)

    def stored_items(self) -> list[dict]:
        return [{"key": item_slug(i.name), "name": i.name, "tracked": True, "rows": i.rows, "first": i.first_ts, "last": i.last_ts}
                for i in self.list_items()]

    def browse(self, key: str, start=None, end=None, page: int = 1, page_size: int = 100) -> tuple[list[dict], int]:
        name = self._name(key)
        if name is None:
            return [], 0
        rows, total = self.fetch(name, ListingQuery(start=start, end=end, page=page, page_size=page_size))
        return [{"id": r["item_id"], **{k: r[k] for k in ("quantity", "price_per_unit", "price", "has_sold", "created_at")}}
                for r in rows], total

    def delete(self, key: str, start=None, end=None, ids: list[str] | None = None) -> int:
        name = self._name(key)
        if name is None:
            return 0
        where, params = self._where(start, end)
        if ids:
            where += (" AND " if where else " WHERE ") + "item_id::text = ANY(%s)"
            params.append(list(ids))
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {}" + where).format(sql.Identifier(table_name(name))), params)
            return int(cur.rowcount)
