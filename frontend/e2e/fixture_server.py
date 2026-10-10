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
        "patch": "Early Access Season #11", "synced_at": "2026-10-10T00:00:00+00:00",
        "trade_totals": {"id.item.test_ore": 1200, "id.item.arming_sword": 95000, "id.item.gold_coins": 0},
        "trade_check_complete": True, "trade_checked_at": "2026-10-10T00:20:00+00:00",
        "items": [
            {"id": "id.item.test_ore_1001", "archetype": "id.item.test_ore", "name": "Test Ore", "rarity": "common",
             "item_type": "misc", "num_primary_attributes": 0, "num_secondary_attributes": 0, "vendor_price": 3},
            {"id": "id.item.bandage_1001", "archetype": "id.item.bandage", "name": "Bandage", "rarity": "poor",
             "item_type": "utility", "num_secondary_attributes": 0, "duration": 4},
            {"id": "id.item.arming_sword_2001", "archetype": "id.item.arming_sword", "name": "Arming Sword", "rarity": "common",
             "item_type": "weapon", "num_secondary_attributes": 0},
            {"id": "id.item.arming_sword_6001", "archetype": "id.item.arming_sword", "name": "Arming Sword", "rarity": "legendary",
             "item_type": "weapon", "num_secondary_attributes": 116, "primary_attributes": [{"attribute": "weapon_damage", "min": 32, "max": 32}]},
            {"id": "id.item.gold_coins", "archetype": "id.item.gold_coins", "name": "Gold Coin", "rarity": "unique",
             "item_type": "misc", "num_secondary_attributes": 0}]}))
    return root


if __name__ == "__main__":
    s = Settings(project_root=build_project(), db_user="", darker_market_api_key="e2e-dummy-key", port=PORT, _env_file=None)
    uvicorn.run(create_app(s), host="127.0.0.1", port=PORT, log_level="warning")
