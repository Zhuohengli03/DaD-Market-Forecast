from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from tests.forecast.conftest import SEASON_START, make_listings


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    make_listings(hours=200, seed=0).to_csv(tmp_path / "test_ore.csv", index=False)
    (tmp_path / "items_config.json").write_text(json.dumps({
        "items": [{"name": "Test Ore", "file": "Test_Ore_API.py", "csv": "test_ore.csv",
                   "category": "ore", "description": "test", "enabled": True}],
        "categories": {"ore": "矿石类"},
    }), encoding="utf-8")
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": [
        {"id": "s1", "name": "Season 1", "start": "2025-10-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "note": ""},
        {"id": "s2", "name": "Season 2", "start": SEASON_START.isoformat(), "end": None, "note": ""},
    ]}), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "Test_Ore_API.py").write_text(
        "import time\nfor i in range(3):\n    print('line', i, flush=True)\n    time.sleep(0.05)\n")
    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> Settings:
    return Settings(project_root=project_root, db_user="", db_password="", darker_market_api_key="", _env_file=None)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app, headers={"X-Darker-Market": "1"}) as c:
        yield c
