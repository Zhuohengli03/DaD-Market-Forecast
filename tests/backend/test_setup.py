import json
import os
import stat

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.envfile import upsert_env
from backend.settings import Settings
from src.api.darkerdb_client import DarkerDBError


def test_first_start_seeds_data_dir(project_root, settings):
    settings.bootstrap()
    data = project_root / "data"
    assert json.loads((data / "items.json").read_text())["items"][0]["name"] == "Test Ore"
    assert [s["id"] for s in json.loads((data / "seasons.json").read_text())["seasons"]] == ["s1", "s2"]
    assert (data / "runs").is_dir()


def test_bootstrap_without_legacy_files_creates_empty_seasons(tmp_path):
    (tmp_path / "items_config.json").write_text('{"items": []}')
    s = Settings(project_root=tmp_path, _env_file=None)
    s.bootstrap()
    assert json.loads((tmp_path / "data" / "seasons.json").read_text()) == {"seasons": []}


def test_bootstrap_never_overwrites_user_state(project_root, settings):
    settings.bootstrap()
    (project_root / "data" / "items.json").write_text('{"items": [], "categories": {}}')
    settings.bootstrap()
    assert json.loads((project_root / "data" / "items.json").read_text())["items"] == []


def test_upsert_env_preserves_other_lines_and_is_private(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DB_USER=me\nDARKER_MARKET_API_KEY=old\n# comment\n")
    upsert_env(env, "DARKER_MARKET_API_KEY", "new-key")
    assert env.read_text() == "DB_USER=me\nDARKER_MARKET_API_KEY=new-key\n# comment\n"
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600
    upsert_env(tmp_path / "fresh.env", "A", "b")
    assert (tmp_path / "fresh.env").read_text() == "A=b\n"


def test_setup_reports_missing_key(client):
    d = client.get("/api/setup").json()["data"]
    assert d["api_key_configured"] is False and d["api_key_hint"] is None
    assert d["seasons"] == 2 and d["tracked_items"] == 1 and d["data_source"] == "csv"


def _app(project_root, validator):
    app = create_app(Settings(project_root=project_root, db_user="", darker_market_api_key="", _env_file=None))
    app.state.key_validator = validator
    return app


def test_put_valid_key_writes_env(project_root):
    seen = []
    app = _app(project_root, lambda k: seen.append(k))
    with TestClient(app) as c:
        app.state.key_validator = lambda k: seen.append(k)
        r = c.put("/api/setup/api-key", json={"api_key": "  abcd1234wxyz  "})
        assert r.status_code == 200 and r.json()["data"]["api_key_hint"] == "…wxyz"
        assert seen == ["abcd1234wxyz"]
        assert "DARKER_MARKET_API_KEY=abcd1234wxyz" in (project_root / ".env").read_text()
        assert c.get("/api/setup").json()["data"]["api_key_configured"] is True


def test_put_rejected_key_is_422_and_not_saved(project_root):
    def bad(k):
        raise DarkerDBError(401, "A valid DarkerDB API key is required")
    app = _app(project_root, bad)
    with TestClient(app) as c:
        app.state.key_validator = bad
        r = c.put("/api/setup/api-key", json={"api_key": "nope"})
        assert r.status_code == 422 and "DarkerDB" in r.json()["error"]
    assert not (project_root / ".env").exists()


def test_put_key_network_failure_is_502(project_root):
    def down(k):
        raise OSError("connection refused")
    app = _app(project_root, down)
    with TestClient(app) as c:
        app.state.key_validator = down
        assert c.put("/api/setup/api-key", json={"api_key": "k"}).status_code == 502


def test_put_empty_key_is_422(client):
    assert client.put("/api/setup/api-key", json={"api_key": "   "}).status_code == 422
