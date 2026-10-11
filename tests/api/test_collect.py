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


def test_main_without_postgres_writes_builtin_db_and_dedupes(tmp_path, monkeypatch):
    import sqlite3
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
    con = sqlite3.connect(tmp_path / "market.db")
    assert [r[0] for r in con.execute("SELECT item_id FROM listings WHERE item='iron_ore' ORDER BY item_id")] == ["1", "2"]


def test_main_reports_missing_key(tmp_path, monkeypatch, capsys):
    import src.api.collect as col
    monkeypatch.setenv("DARKER_MARKET_API_KEY", "")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)]) == 2
    assert "设置" in capsys.readouterr().out


def test_main_is_incremental_from_latest_stored(tmp_path, monkeypatch):
    import src.api.collect as col
    from src.api.storage import SqliteStorage
    SqliteStorage(tmp_path / "market.db", "iron_ore").insert([_row(1)])
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


def _season_file(tmp_path, start="2026-10-09T00:00:00+00:00"):
    import json
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": [
        {"id": "s10", "name": "第 10 赛季", "start": "2026-07-17T00:00:00+00:00", "end": start},
        {"id": "s11", "name": "第 11 赛季", "start": start, "end": None}]}))


def _fake(seen):
    class FakeClient:
        def __init__(self, key):
            pass

        def find_item_id(self, name):
            return "id.item.iron_ores_2001"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None, until=None):
            seen.update(since=since, until=until, sort=sort)
            yield []
    return FakeClient


def test_collector_never_backfills_an_older_season(tmp_path, monkeypatch):
    """Stored data stops in an earlier season: resume from the current season's start, not from that old listing."""
    import src.api.collect as col
    from src.api.storage import SqliteStorage
    SqliteStorage(tmp_path / "market.db", "iron_ore").insert([_row(1)])      # 2026-04-04
    _season_file(tmp_path)
    seen = {}
    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", _fake(seen))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path)])
    assert seen["since"] == "2026-10-09T00:00:00+00:00" and seen["until"] is None


def test_collector_collects_one_season_between_given_times(tmp_path, monkeypatch):
    import src.api.collect as col
    seen = {}
    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", _fake(seen))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    col.main(["--item", "Iron Ore", "--data-dir", str(tmp_path), "--since", "2026-07-17T00:00:00+00:00",
              "--until", "2026-10-09T00:00:00+00:00"])
    assert seen == {"since": "2026-07-17T00:00:00+00:00", "until": "2026-10-09T00:00:00+00:00", "sort": "created_at:asc"}


def test_collector_records_how_far_a_season_was_collected(tmp_path, monkeypatch):
    """A page budget can stop a long backfill midway; the next run must continue from there, not restart."""
    import json
    import src.api.collect as col

    pages = [[{**_raw(i)} for i in range(k * 3, k * 3 + 3)] for k in range(4)]

    class FakeClient:
        credits_used = 0

        def __init__(self, key):
            pass

        def find_item_id(self, name):
            return "id.item.iron_ores_2001"

        def iter_market(self, item_id, since=None, limit=50, sort="created_at:desc", max_pages=None, until=None):
            yield from pages[:max_pages]

    monkeypatch.setenv("DARKER_MARKET_API_KEY", "k")
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setattr(col, "DarkerDBClient", FakeClient)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    args = ["--item", "Iron Ore", "--data-dir", str(tmp_path), "--since", "2026-04-04T00:00:00Z", "--until", "2026-04-05T00:00:00Z", "--coverage-key", "s10"]
    col.main(args + ["--max-pages", "2"])                         # stopped by the page budget
    marker = json.loads((tmp_path / "coverage" / "iron_ore.json").read_text())
    assert marker["s10"] == "2026-04-04T14:05:00Z"                 # newest listing seen so far
    col.main(args + ["--max-pages", "10"])                        # ran out of pages: the whole window is done
    marker = json.loads((tmp_path / "coverage" / "iron_ore.json").read_text())
    assert marker["s10"] == "2026-04-05T00:00:00Z"


def _raw(i):
    return {"id": str(100 + i), "item_id": "id.item.iron_ores_2001", "name": "Iron Ore", "quantity": 2, "price": 170, "price_per_unit": 85,
            "has_sold": False, "created_at": f"2026-04-04T14:{i:02d}:00Z"}
