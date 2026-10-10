import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.data.items import ItemsStore
from backend.main import create_app
from backend.services.envfile import upsert_env
from backend.settings import Settings

H = {"X-Darker-Market": "1"}


def _app(project_root, **kw):
    return create_app(Settings(project_root=project_root, db_user="", darker_market_api_key=kw.pop("key", ""), _env_file=None, **kw))


# ---- I1 cross-site requests -------------------------------------------------
def test_mutations_without_app_header_are_refused(project_root):
    with TestClient(_app(project_root)) as c:
        assert c.post("/api/scheduler/stop", headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 403
        assert c.post("/api/items/track", content=b'{"item_id":"id.item.x"}').status_code == 403
        assert c.post("/api/scheduler/stop", headers=H).status_code == 200
        assert c.get("/api/status").status_code == 200          # reads stay open


def test_foreign_host_header_is_refused(project_root):
    with TestClient(_app(project_root), base_url="http://evil.example") as c:
        assert c.get("/api/status").status_code == 400
    with TestClient(_app(project_root), base_url="http://localhost:5001") as c:
        assert c.get("/api/status").status_code == 200


def test_lan_mode_accepts_other_hosts(project_root):
    with TestClient(_app(project_root, host="0.0.0.0"), base_url="http://192.168.1.5:5001") as c:
        assert c.get("/api/status").status_code == 200


# ---- I2 / I3 names with punctuation or leading digits -----------------------
def test_forecast_for_item_with_apostrophe_saves(project_root):
    import pandas as pd
    from src.api.storage import SqliteStorage
    from src.api.darkerdb_client import canonical_ts
    app = _app(project_root)
    with TestClient(app, headers=H) as c:
        df = pd.read_csv(project_root / "test_ore.csv", dtype={"item_id": str})
        SqliteStorage(app.state.settings.sqlite_path, "adventurer_s_boots").insert([
            {"id": r.item_id, "item": "Adventurer's Boots", "quantity": r.quantity, "price_per_unit": r.price_per_unit,
             "price": r.price, "has_sold": bool(r.has_sold), "created_at": canonical_ts(r.created_at)} for r in df.itertuples()])
        cfg = c.get("/api/config/items").json()["data"]
        cfg["items"].append({"name": "Adventurer's Boots", "enabled": True, "category": "", "description": ""})
        c.put("/api/config/items", json=cfg)
        job_id = c.post("/api/forecast", json={"item": "Adventurer's Boots", "season_id": "s2", "horizon_hours": 24, "models": ["ridge"]}).json()["data"]["job_id"]
        for _ in range(600):
            j = c.get(f"/api/jobs/{job_id}").json()["data"]
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert j["status"] == "succeeded", j


def test_digit_leading_item_collected_csv_is_found(project_root):
    from src.api.storage import SqliteStorage, item_slug
    assert item_slug("3 Leaf Clover") == "item_3_leaf_clover"
    app = _app(project_root)
    with TestClient(app, headers=H) as c:
        SqliteStorage(app.state.settings.sqlite_path, item_slug("3 Leaf Clover")).insert(
            [{"id": "1", "item": "3 Leaf Clover", "quantity": 1, "price_per_unit": 2.0, "price": 2.0, "has_sold": False, "created_at": "2026-01-20T01:00:00Z"}])
        cfg = c.get("/api/config/items").json()["data"]
        cfg["items"].append({"name": "3 Leaf Clover", "enabled": True, "category": "", "description": ""})
        c.put("/api/config/items", json=cfg)
        rows = {i["name"]: i["rows"] for i in c.get("/api/items").json()["data"]}
        assert rows["3 Leaf Clover"] == 1


# ---- I5 scheduler restarts when tracked items change --------------------------
def test_scheduler_restarts_after_item_change(project_root):
    (project_root / "src" / "scheduler").mkdir(exist_ok=True)
    (project_root / "src" / "scheduler" / "start_scheduler.py").write_text("import time\nwhile True: time.sleep(0.1)\n")
    app = _app(project_root)
    with TestClient(app, headers=H) as c:
        c.post("/api/scheduler/start")
        first = app.state.scheduler._proc
        cfg = c.get("/api/config/items").json()["data"]
        cfg["items"][0]["schedule"] = "06:00"
        c.put("/api/config/items", json=cfg)
        assert app.state.scheduler._proc is not first and app.state.scheduler.running()
        first.wait(timeout=5)


def test_scheduler_runs_one_collection_at_a_time():
    from src.scheduler.task_config import SCHEDULER_CONFIG
    assert SCHEDULER_CONFIG["max_concurrent_tasks"] == 1


# ---- I6 + M1 + M2 API key handling ------------------------------------------
def test_key_errors_are_bilingual_and_never_echo_the_key(project_root):
    from src.api.darkerdb_client import DarkerDBError
    app = _app(project_root)
    with TestClient(app, headers=H) as c:
        def bad(k):
            raise DarkerDBError(401, "nope")
        app.state.key_validator = bad
        msg = c.put("/api/setup/api-key", json={"api_key": "secretkey123"}).json()["error"]
        assert "DarkerDB" in msg and "rejected" in msg and "secretkey123" not in msg

        def down(k):
            raise OSError("failed for header X-API-Key: secretkey123")
        app.state.key_validator = down
        r = c.put("/api/setup/api-key", json={"api_key": "secretkey123"})
        assert r.status_code == 502 and "secretkey123" not in r.text and "network" in r.text.lower()


def test_key_with_newline_is_rejected(client):
    r = client.put("/api/setup/api-key", json={"api_key": "abc\nDB_USER=evil"}, headers=H)
    assert r.status_code == 422


def test_upsert_env_refuses_newlines(tmp_path):
    with pytest.raises(ValueError):
        upsert_env(tmp_path / ".env", "K", "a\nB=c")


def test_default_validator_checks_market_scope(monkeypatch):
    import backend.api.setup as setup
    paths = []

    class C:
        def __init__(self, *a, **k):
            pass

        def get(self, path, params=None):
            paths.append(path)
            return {"body": []}
    monkeypatch.setattr(setup, "DarkerDBClient", C)
    setup.default_key_validator("k")
    assert paths == ["/items", "/market"]


# ---- M3 rarity variants -------------------------------------------------------
def test_same_name_variants_get_distinct_names(tmp_path):
    s = ItemsStore(tmp_path / "items.json")
    a = s.track("Arming Sword", "id.item.arming_sword_2001", description="uncommon")
    b = s.track("Arming Sword", "id.item.arming_sword_3001", description="rare")
    assert a["name"] == "Arming Sword" and b["name"] == "Arming Sword (rare)"
    with pytest.raises(ValueError):
        s.track("Arming Sword", "id.item.arming_sword_3001", description="rare")


# ---- M4 / M5 CSV integrity ----------------------------------------------------
def test_csv_reader_merges_legacy_and_data_files_and_drops_duplicates(project_root):
    from backend.data.csv_repo import CsvListingRepository
    s = Settings(project_root=project_root, _env_file=None)
    s.bootstrap()
    legacy = (project_root / "test_ore.csv").read_text().splitlines()
    (project_root / "data" / "test_ore.csv").write_text("\n".join([legacy[0], legacy[1], legacy[1], legacy[2]]) + "\n")
    repo = CsvListingRepository(s.csv_dirs, s.items_config_path)
    assert repo.list_items()[0].rows == 1000


def test_csv_storage_lock_serialises_writers(tmp_path):
    import threading
    from src.api.storage import CsvStorage
    path = tmp_path / "x.csv"
    rows = [{"id": str(i), "item": "X", "quantity": 1, "price_per_unit": 1.0, "price": 1.0, "has_sold": False,
             "created_at": f"2026-01-01T00:00:{i:02d}Z"} for i in range(50)]
    ts = [threading.Thread(target=lambda: CsvStorage(path).insert(rows)) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(path.read_text().strip().splitlines()) == 51


# ---- M6 / M7 catalog robustness -------------------------------------------------
def test_patch_lookup_with_unexpected_body_does_not_abort(settings):
    from backend.data.catalog import CatalogStore
    from backend.services.catalog_sync import sync_catalog

    class C:
        def get(self, path, params=None):
            return {"body": ["0.17"]}

        def iter_pages(self, path, params, max_pages=None):
            yield [{"id": "id.item.a", "name": "A"}]
    settings.bootstrap()
    assert sync_catalog(C(), CatalogStore(settings.catalog_path), log=lambda s: None) == 1


def test_catalog_store_caches_parsed_file(settings, monkeypatch):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    store.save_items([{"id": "id.item.a", "name": "A"}], None)
    loads = []
    real = json.loads
    monkeypatch.setattr("backend.data.catalog.json.loads", lambda s: loads.append(1) or real(s))
    for _ in range(5):
        store.query("a")
    assert len(loads) <= 1


def test_tracking_upgrades_a_name_only_entry_instead_of_duplicating(tmp_path):
    s = ItemsStore(tmp_path / "items.json")
    s.write({"items": [{"name": "Iron Ore", "category": "ore", "description": "铁矿石", "enabled": True}], "categories": {}})
    entry = s.track("Iron Ore", "id.item.iron_ores_2001", "misc", "common")
    assert entry["name"] == "Iron Ore" and entry["item_id"] == "id.item.iron_ores_2001"
    items = s.list()
    assert len(items) == 1 and items[0]["item_id"] == "id.item.iron_ores_2001" and items[0]["description"] == "铁矿石"
    import pytest
    with pytest.raises(ValueError):
        s.track("Iron Ore", "id.item.iron_ores_2001")
