import csv

from src.api.storage import CsvStorage


def _row(i, ts):
    return {"id": str(i), "item": "Iron Ore", "quantity": 6, "price_per_unit": 85.0, "price": 510.0,
            "has_sold": True, "created_at": ts}


def test_csv_storage_appends_and_dedupes(tmp_path):
    path = tmp_path / "iron_ore.csv"
    s = CsvStorage(path)
    assert s.existing_keys() == set()
    assert s.insert([_row(1, "2026-01-01T00:00:00Z"), _row(2, "2026-01-01T00:01:00Z")]) == 2
    s2 = CsvStorage(path)
    assert s2.existing_keys() == {"1_2026-01-01T00:00:00Z", "2_2026-01-01T00:01:00Z"}
    assert s2.insert([_row(2, "2026-01-01T00:01:00Z"), _row(3, "2026-01-01T00:02:00Z")]) == 1
    rows = list(csv.DictReader(path.open()))
    assert [r["item_id"] for r in rows] == ["1", "2", "3"]
    assert set(rows[0]) == {"item_id", "item", "quantity", "price_per_unit", "price", "has_sold", "created_at", "insert_time"}
