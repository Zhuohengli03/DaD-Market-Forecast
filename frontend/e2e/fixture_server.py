"""Backend on synthetic data for Playwright: no database, no network, no real API key needed.

    python frontend/e2e/fixture_server.py      (run from the project root)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from backend.main import create_app  # noqa: E402
from backend.settings import Settings  # noqa: E402
from tests.forecast.conftest import SEASON_START, make_listings  # noqa: E402

PORT = 5002


def build_project() -> Path:
    root = Path(tempfile.mkdtemp(prefix="darker-e2e-"))
    data = root / "data"
    (data / "runs").mkdir(parents=True)
    make_listings(hours=240, seed=0).to_csv(data / "test_ore.csv", index=False)
    (data / "items.json").write_text(json.dumps({"items": [
        {"name": "Test Ore", "item_id": "id.item.test_ore_1001", "category": "Misc", "description": "fixture", "enabled": True}],
        "categories": {}}))
    (data / "seasons.json").write_text(json.dumps({"seasons": [
        {"id": "s-e2e", "name": "Season 2", "start": SEASON_START.isoformat(), "end": None, "note": ""}]}))
    (data / "catalog.json").write_text(json.dumps({
        "patch": "0.0.0-e2e", "synced_at": "2026-10-01T00:00:00+00:00", "market_scanned_at": "2026-10-01T00:00:00+00:00",
        "market_since": SEASON_START.isoformat(),
        "items": [{"id": "id.item.test_ore_1001", "name": "Test Ore", "rarity": "common", "item_type": "Misc", "vendor_price": 3},
                  {"id": "id.item.arming_sword_2001", "name": "Arming Sword", "rarity": "uncommon", "slot_type": "Weapon",
                   "primary": {"weapon_damage": 30}}],
        "market": {"id.item.test_ore_1001": {"name": "Test Ore", "listings": 50, "sold": 40, "min_price_per_unit": 15,
                                             "median_price_per_unit": 21, "last_seen": "2026-01-30T00:00:00Z"}}}))
    return root


if __name__ == "__main__":
    s = Settings(project_root=build_project(), db_user="", darker_market_api_key="e2e-dummy-key", port=PORT, _env_file=None)
    uvicorn.run(create_app(s), host="127.0.0.1", port=PORT, log_level="warning")
