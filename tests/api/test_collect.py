import json
from src.api.collect import collect_pages, parse_args


class _Client:
    def __init__(self, pages):
        self._pages = pages
        self.kwargs = None

    def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None):
        self.kwargs = dict(item_id=item_id, since=since, max_pages=max_pages)
        for p in self._pages[: max_pages or len(self._pages)]:
            yield p


def _row(i, qty=6):
    return {"id": str(i), "item": "Iron Ore", "quantity": qty, "price": 510.0, "price_per_unit": 85.0,
            "has_sold": True, "created_at": f"2026-04-04T14:{i:02d}:00Z"}


def test_collect_pages_dedups_and_stops_after_empty_streak():
    pages = [[_row(1), _row(2)], [_row(2), _row(3)], [_row(3)], [_row(3)], [_row(3)], [_row(9)]]
    client = _Client(pages)
    seen = {"2_2026-04-04T14:02:00Z"}
    logs = []
    rows = collect_pages(client, "id.item.iron_ore_1001", "Iron Ore", since="2026-04-01", seen=seen,
                         max_pages=200, max_empty_pages=3, log=logs.append)
    assert [r["id"] for r in rows] == ["1", "3"]          # 2 was already known, 9 never reached
    assert client.kwargs["since"] == "2026-04-01"
    assert any("停止" in l for l in logs)


def test_collect_pages_normalizes_and_skips_bad_rows():
    pages = [[{"id": "1", "item": {"name": "Iron Ore"}, "quantity": 2, "price": 10, "created_at": "2026-04-04T14:01:00Z"},
              {"id": "bad"}]]
    logs = []
    rows = collect_pages(_Client(pages), "x", "Iron Ore", since=None, seen=set(), max_pages=5, log=logs.append)
    assert rows == [{"id": "1", "item": "Iron Ore", "quantity": 2, "price_per_unit": 5.0, "price": 10.0,
                     "has_sold": False, "created_at": "2026-04-04T14:01:00Z"}]
    assert any("跳过" in l for l in logs)


def test_parse_args_defaults():
    a = parse_args(["--item", "Gold Ore"])
    assert a.item == "Gold Ore" and a.max_pages == 200 and a.since is None and not a.probe


def test_main_without_postgres_writes_csv_and_dedupes(tmp_path, monkeypatch):
    import csv as _csv
    import src.api.collect as col

    class FakeClient:
        def __init__(self, key):
            assert key == "k"

        def find_item_id(self, name):
            return "id.item.iron_ore_1001"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None):
            yield [_row(1), _row(2)]

    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", FakeClient)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)]) == 0
    assert col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)]) == 0
    rows = list(_csv.DictReader((tmp_path / "iron_ore.csv").open()))
    assert [r["item_id"] for r in rows] == ["1", "2"]


def test_main_reports_missing_key(tmp_path, monkeypatch, capsys):
    import src.api.collect as col
    monkeypatch.setenv("DARKER_MARKET_API_KEY", "")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)]) == 2
    assert "设置" in capsys.readouterr().out


def test_main_is_incremental_from_latest_stored(tmp_path, monkeypatch):
    import src.api.collect as col
    from src.api.storage import CsvStorage
    CsvStorage(tmp_path / "iron_ore.csv").insert([_row(1)])
    seen = {}

    class FakeClient:
        def __init__(self, key):
            pass

        def find_item_id(self, name):
            return "id.item.iron_ores_2001"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None):
            seen["since"], seen["sort"] = since, sort
            yield []

    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", FakeClient)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)])
    assert seen["since"] == _row(1)["created_at"]
    assert seen["sort"] == "created_at:asc"     # resume oldest-first so a page budget never leaves a gap


def test_collector_emits_progress_lines_when_asked(tmp_path, monkeypatch, capsys):
    import src.api.collect as col

    class FakeClient:
        def __init__(self, key):
            pass

        def find_item_id(self, name):
            return "id.item.iron_ores_2001"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None):
            yield [_row(1), _row(2)]

    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.setenv("DARKER_PROGRESS", "1")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", FakeClient)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path), "--max-pages", "5"])
    lines = [json.loads(l[2:]) for l in capsys.readouterr().out.splitlines() if l.startswith("@@")]
    assert lines[0]["stage"] == "collect" and lines[0]["done"] == 1 and lines[0]["total"] == 5
    assert lines[-1] == {"stage": "done", "new": 2}
