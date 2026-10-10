import pandas as pd
import pytest

from backend.data.csv_repo import CsvListingRepository
from backend.data.repository import ListingQuery, make_repository, table_name, validate_item
from backend.settings import Settings


def test_validate_item_and_table_name():
    assert table_name("Iron Ore") == "iron_ore"
    assert table_name("2H Sword") == "item_2h_sword" and table_name("Lich's Hood") == "lich_s_hood"
    for bad in ("", "x" * 81, "drop; table", "a\nb"):
        with pytest.raises(ValueError):
            validate_item(bad)


def test_csv_list_items(project_root):
    repo = CsvListingRepository(project_root, project_root / "items_config.json")
    items = repo.list_items()
    assert items[0].name == "Test Ore" and items[0].rows == 1000
    assert items[0].first_ts.endswith("+00:00")


def test_csv_fetch_filters_and_pages(project_root):
    repo = CsvListingRepository(project_root, project_root / "items_config.json")
    rows, total = repo.fetch("Test Ore", ListingQuery(page=1, page_size=10))
    assert len(rows) == 10 and total == 1000
    assert rows[0]["created_at"] >= rows[1]["created_at"]  # newest first
    rows2, _ = repo.fetch("Test Ore", ListingQuery(page=2, page_size=10))
    assert rows2[0]["item_id"] != rows[0]["item_id"]
    sold_rows, sold_total = repo.fetch("Test Ore", ListingQuery(sold=True, page_size=500))
    assert all(r["has_sold"] for r in sold_rows) and sold_total < 1000
    qty_rows, _ = repo.fetch("Test Ore", ListingQuery(min_qty=6, max_qty=6, page_size=500))
    assert all(r["quantity"] == 6 for r in qty_rows)
    start = pd.Timestamp("2026-01-25T00:00:00Z")
    _, n_after = repo.fetch("Test Ore", ListingQuery(start=start, page_size=1))
    assert 0 < n_after < 1000


def test_csv_fetch_all_range(project_root):
    repo = CsvListingRepository(project_root, project_root / "items_config.json")
    df = repo.fetch_all("Test Ore", pd.Timestamp("2026-01-21T00:00:00Z"), pd.Timestamp("2026-01-22T00:00:00Z"))
    assert len(df) == 24 * 5
    with pytest.raises(KeyError):
        repo.fetch_all("Ghost", None, None)


def test_make_repository_falls_back_to_builtin_db(project_root, monkeypatch):
    import backend.data.postgres_repo as pg

    def boom(*a, **k):
        raise RuntimeError("refused")
    monkeypatch.setattr(pg.psycopg2, "connect", boom)
    s = Settings(project_root=project_root, db_user="u", db_password="p", db_host="127.0.0.1")
    logs = []
    repo = make_repository(s, log=logs.append)
    assert repo.source_name() == "sqlite" and any("refused" in l for l in logs)


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []
        self.description = [("item_id",), ("item",), ("quantity",), ("price_per_unit",), ("price",),
                            ("has_sold",), ("created_at",)]

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self, **kw):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_postgres_where_casts_varchar_created_at():
    from backend.data.postgres_repo import PostgresListingRepository
    where, params = PostgresListingRepository._where(pd.Timestamp("2026-01-01T00:00:00Z"), None, sold=True)
    assert "created_at::timestamptz >= %s" in where and "has_sold = %s" in where
    assert len(params) == 2


def test_postgres_fetch_all_without_sqlalchemy(project_root, monkeypatch):
    from backend.data.postgres_repo import PostgresListingRepository
    s = Settings(project_root=project_root, db_user="u", db_password="p")
    repo = PostgresListingRepository(s)
    cur = _FakeCursor([("1", "Gold Ore", 6, 20.0, 120.0, True, "2026-01-29T12:32:27Z")])
    monkeypatch.setattr(repo, "_conn", lambda: _FakeConn(cur))
    df = repo.fetch_all("Gold Ore", pd.Timestamp("2026-01-01T00:00:00Z"), None)
    assert list(df.columns) == ["item_id", "item", "quantity", "price_per_unit", "price", "has_sold", "created_at"]
    assert str(df["created_at"].dt.tz) == "UTC" and len(df) == 1
    assert "ORDER BY created_at::timestamptz" in repr(cur.executed[-1][0])


def test_postgres_list_items_accepts_tz_aware_datetimes(project_root, monkeypatch):
    from datetime import datetime, timezone
    from backend.data.postgres_repo import PostgresListingRepository

    class _Cur(_FakeCursor):
        def __init__(self):
            super().__init__([])
            self._answers = [("test_ore",), (5, datetime(2026, 1, 20, tzinfo=timezone.utc), datetime(2026, 1, 29, 12, tzinfo=timezone.utc))]

        def fetchone(self):
            return self._answers.pop(0)

    s = Settings(project_root=project_root, db_user="u", db_password="p", _env_file=None)
    s.bootstrap()
    repo = PostgresListingRepository(s)
    monkeypatch.setattr(repo, "_conn", lambda: _FakeConn(_Cur()))
    items = repo.list_items()
    assert items[0].rows == 5 and items[0].first_ts == "2026-01-20T00:00:00+00:00"
    assert items[0].last_ts == "2026-01-29T12:00:00+00:00"
