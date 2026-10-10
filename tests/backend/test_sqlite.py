import sqlite3

import pandas as pd
import pytest

from backend.data.repository import ListingQuery, make_repository
from backend.data.sqlite_repo import SqliteListingRepository, import_csv_once
from backend.settings import Settings
from src.api.storage import SqliteStorage, open_storage
from src.api.sqlite_db import connect


def _row(i, ts, qty=6, sold=True, ppu=85.0):
    return {"id": str(i), "item": "Iron Ore", "quantity": qty, "price_per_unit": ppu, "price": ppu * qty,
            "has_sold": sold, "created_at": ts}


def test_schema_is_created_with_unique_key_and_index(tmp_path):
    con = connect(tmp_path / "market.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(listings)")]
    assert cols == ["item", "item_id", "item_name", "quantity", "price_per_unit", "price", "has_sold", "created_at", "insert_time"]
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    idx = [r[1] for r in con.execute("PRAGMA index_list(listings)")]
    assert any("created" in name for name in idx)


def test_storage_inserts_dedupes_and_reports_latest(tmp_path):
    db = tmp_path / "market.db"
    s = SqliteStorage(db, "iron_ore")
    assert s.latest_created_at() is None and s.existing_keys() == set()
    assert s.insert([_row(1, "2026-01-01T00:00:00Z"), _row(2, "2026-01-01T00:05:00Z")]) == 2
    assert s.insert([_row(2, "2026-01-01T00:05:00Z"), _row(3, "2026-01-01T00:09:00Z")]) == 1
    assert s.latest_created_at() == "2026-01-01T00:09:00Z"
    assert SqliteStorage(db, "iron_ore").existing_keys() == {"1_2026-01-01T00:00:00Z", "2_2026-01-01T00:05:00Z", "3_2026-01-01T00:09:00Z"}
    assert SqliteStorage(db, "gold_ore").latest_created_at() is None      # items are kept apart


def test_open_storage_defaults_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_USER", raising=False)
    s = open_storage("iron_ore", tmp_path, log=lambda m: None)
    assert isinstance(s, SqliteStorage) and (tmp_path / "market.db").exists()


def test_repository_reads_filters_and_pages(project_root):
    s = Settings(project_root=project_root, _env_file=None)
    s.bootstrap()
    st = SqliteStorage(s.sqlite_path, "test_ore")
    st.insert([_row(i, f"2026-01-2{i % 3}T0{i % 9}:00:00Z", qty=1 + i % 6, sold=i % 2 == 0) for i in range(30)])
    repo = SqliteListingRepository(s.sqlite_path, s.items_config_path)
    item = repo.list_items()[0]
    assert item.name == "Test Ore" and item.rows == 30 and item.last_ts.endswith("+00:00")
    rows, total = repo.fetch("Test Ore", ListingQuery(page=1, page_size=10))
    assert total == 30 and len(rows) == 10 and rows[0]["created_at"] >= rows[-1]["created_at"]
    sold, n = repo.fetch("Test Ore", ListingQuery(sold=True, min_qty=2, page_size=100))
    assert n == len(sold) and all(r["has_sold"] and r["quantity"] >= 2 for r in sold)
    df = repo.fetch_all("Test Ore", pd.Timestamp("2026-01-21T00:00:00Z"), pd.Timestamp("2026-01-22T00:00:00Z"))
    assert len(df) == 10 and str(df["created_at"].dt.tz) == "UTC"
    with pytest.raises(KeyError):
        repo.fetch_all("Ghost", None, None)


def test_legacy_csv_imported_once(project_root):
    s = Settings(project_root=project_root, _env_file=None)
    s.bootstrap()
    assert import_csv_once(s) == {"Test Ore": 1000}          # fixture writes test_ore.csv in the project root
    assert import_csv_once(s) == {}                           # second time: already imported
    con = sqlite3.connect(s.sqlite_path)
    assert con.execute("SELECT COUNT(*) FROM listings WHERE item='test_ore'").fetchone()[0] == 1000


def test_make_repository_uses_sqlite_without_postgres(project_root):
    s = Settings(project_root=project_root, db_user="", _env_file=None)
    s.bootstrap()
    repo = make_repository(s, log=lambda m: None)
    assert repo.source_name() == "sqlite" and repo.list_items()[0].rows == 1000
