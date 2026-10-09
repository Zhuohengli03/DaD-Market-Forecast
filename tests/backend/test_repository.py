import pandas as pd
import pytest

from backend.data.csv_repo import CsvListingRepository
from backend.data.repository import ListingQuery, make_repository, table_name, validate_item
from backend.settings import Settings


def test_validate_item_and_table_name():
    assert table_name("Iron Ore") == "iron_ore"
    for bad in ("", "x" * 65, "drop; table", "Iron-Ore"):
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


def test_make_repository_falls_back_to_csv(project_root, monkeypatch):
    import backend.data.postgres_repo as pg

    def boom(*a, **k):
        raise RuntimeError("refused")
    monkeypatch.setattr(pg.psycopg2, "connect", boom)
    s = Settings(project_root=project_root, db_user="u", db_password="p", db_host="127.0.0.1")
    logs = []
    repo = make_repository(s, log=logs.append)
    assert repo.source_name() == "csv" and any("refused" in l for l in logs)
