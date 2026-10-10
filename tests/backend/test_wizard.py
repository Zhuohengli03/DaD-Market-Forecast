import json
import sys
import time

from fastapi.testclient import TestClient

from backend.data.app_settings import AppSettings
from backend.main import create_app
from backend.settings import Settings
from src.scheduler.task_config import tasks_from_items

H = {"X-Darker-Market": "1"}


def _wait(c, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_app_settings_defaults_and_update(tmp_path):
    s = AppSettings(tmp_path / "settings.json")
    assert s.get() == {"initialized_at": None, "collect_interval_seconds": 21600, "auto_collect": False}
    s.update(collect_interval_seconds=90, auto_collect=True)
    assert AppSettings(tmp_path / "settings.json").get()["collect_interval_seconds"] == 90


def test_app_settings_migrates_old_hour_setting(tmp_path):
    (tmp_path / "settings.json").write_text('{"collect_interval_hours": 3}')
    assert AppSettings(tmp_path / "settings.json").get()["collect_interval_seconds"] == 10800


def test_setup_reports_environment_and_wizard_state(client):
    d = client.get("/api/setup").json()["data"]
    assert d["initialized"] is False and d["collect_interval_seconds"] == 21600 and d["auto_collect"] is False
    assert d["rate_limit_per_minute"] == 60
    env = d["environment"]
    assert env["python_ok"] is True and env["python_version"].startswith("3.") and env["storage"] == "sqlite"
    assert env["data_dir"].endswith("data")


def test_schedule_endpoint_validates_and_drives_tasks(client, project_root):
    assert client.put("/api/setup/schedule", json={"interval_seconds": 29, "auto_collect": False}).status_code == 422
    assert client.put("/api/setup/schedule", json={"interval_seconds": 31 * 86400, "auto_collect": False}).status_code == 422
    r = client.put("/api/setup/schedule", json={"interval_seconds": 30, "auto_collect": False})
    assert r.status_code == 200 and r.json()["data"]["collect_interval_seconds"] == 30
    tasks = client.get("/api/scheduler").json()["data"]["tasks"]
    assert tasks[0]["schedule_type"] == "seconds" and tasks[0]["schedule_value"] == "30"


def test_tasks_use_interval_from_settings_file(tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [{"name": "Iron Ore", "item_id": "id.item.iron_ores_2001", "enabled": True}]}))
    (tmp_path / "settings.json").write_text(json.dumps({"collect_interval_seconds": 93784}))
    t = tasks_from_items(items, settings_path=tmp_path / "settings.json")
    assert t[0]["schedule_type"] == "seconds" and t[0]["schedule_value"] == 93784
    assert t[0]["description"] == "每 1天2时3分4秒 采集 Iron Ore"
    assert t[0]["args"][-2:] == ["--item-id", "id.item.iron_ores_2001"]


def test_auto_collect_starts_scheduler_and_on_next_launch(project_root):
    (project_root / "src" / "scheduler").mkdir(exist_ok=True)
    (project_root / "src" / "scheduler" / "start_scheduler.py").write_text("import time\nwhile True: time.sleep(0.1)\n")
    mk = lambda: create_app(Settings(project_root=project_root, db_user="", darker_market_api_key="", _env_file=None))
    app = mk()
    with TestClient(app, headers=H) as c:
        c.put("/api/setup/schedule", json={"interval_seconds": 21600, "auto_collect": True})
        assert app.state.scheduler.running()
    app2 = mk()
    with TestClient(app2, headers=H):
        assert app2.state.scheduler.running()          # resumed automatically on launch
    with TestClient(mk(), headers=H) as c:
        c.put("/api/setup/schedule", json={"interval_seconds": 21600, "auto_collect": False})


def test_database_setup_tests_connection_and_falls_back(client, monkeypatch, project_root):
    import backend.api.setup as setup_api

    def bad(**kw):
        raise OSError("connection refused")
    monkeypatch.setattr(setup_api, "probe_postgres", bad)
    r = client.put("/api/setup/database", json={"host": "localhost", "port": 5432, "database": "d", "user": "u", "password": "p"})
    assert r.status_code == 422 and "连接" in r.json()["error"]
    assert not (project_root / ".env").exists()

    monkeypatch.setattr(setup_api, "probe_postgres", lambda **kw: None)
    real = setup_api.make_repository
    fake_pg = type("R", (), {"source_name": lambda self: "postgres", "list_items": lambda self: []})()
    monkeypatch.setattr(setup_api, "make_repository", lambda s, log=print: fake_pg if s.db_user else real(s, log))
    r = client.put("/api/setup/database", json={"host": "localhost", "port": 5432, "database": "d", "user": "u", "password": "p"})
    assert r.status_code == 200 and r.json()["data"]["environment"]["storage"] == "postgres"
    env = (project_root / ".env").read_text()
    assert "DB_USER=u" in env and "DB_PASSWORD=p" in env
    r = client.delete("/api/setup/database")
    assert r.json()["data"]["environment"]["storage"] == "sqlite" and "DB_USER=\n" in (project_root / ".env").read_text()


def test_initialize_runs_every_step_and_reports_per_item(client, project_root, settings):
    from backend.data.catalog import CatalogStore
    store = CatalogStore(settings.catalog_path)
    store.save_items([{"id": "id.item.test_ore_1001", "archetype": "id.item.test_ore", "name": "Test Ore", "rarity": "common",
                       "item_type": "misc", "num_secondary_attributes": 0}], "Early Access Season #11")
    script = project_root / "fake_collect.py"
    script.write_text('import json\nprint("@@" + json.dumps({"stage": "collect", "done": 1, "total": 5, "new": 40}), flush=True)\n'
                      'print("@@" + json.dumps({"stage": "done", "new": 40}), flush=True)\n')
    client.app.state.settings.darker_market_api_key = "k"
    client.app.state.collector_argv = lambda entry, since: [sys.executable, str(script)]
    r = client.post("/api/setup/initialize", json={"item_ids": ["id.item.test_ore_1001"], "max_pages": 5}, headers=H)
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded", job
    detail = job["progress"]["detail"]
    assert job["progress"]["stage"] == "done"
    row = detail["items"][0]
    assert row["name"] == "Test Ore" and row["collected"] == 40 and row["forecast"] in ("ok", "not_enough_data", "failed")
    assert client.get("/api/setup").json()["data"]["initialized"] is True
    assert any(i["name"] == "Test Ore" for i in client.get("/api/items").json()["data"])


def test_initialize_requires_key_and_items(client):
    client.app.state.settings.darker_market_api_key = ""
    assert client.post("/api/setup/initialize", json={"item_ids": ["id.item.x_1001"]}, headers=H).status_code == 409
    client.app.state.settings.darker_market_api_key = "k"
    assert client.post("/api/setup/initialize", json={"item_ids": []}, headers=H).status_code == 422
