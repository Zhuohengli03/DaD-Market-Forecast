import json
import subprocess
import sys
from pathlib import Path

from tests.forecast.conftest import SEASON_START, make_listings

ROOT = Path(__file__).resolve().parents[2]


def test_cli_writes_json(tmp_path: Path):
    df = make_listings(hours=200)
    df.to_csv(tmp_path / "test_ore.csv", index=False)
    (tmp_path / "items_config.json").write_text(json.dumps({"items": [
        {"name": "Test Ore", "file": "x.py", "csv": "test_ore.csv", "category": "ore", "enabled": True}]}))
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": [
        {"id": "s2", "name": "S2", "start": SEASON_START.isoformat(), "end": None}]}))
    out = tmp_path / "run.json"
    proc = subprocess.run([sys.executable, "-m", "src.forecast.cli", "--item", "Test Ore", "--season", "s2",
                           "--horizon", "24", "--models", "ridge", "--csv-dir", str(tmp_path),
                           "--seasons", str(tmp_path / "seasons.json"), "--out", str(out)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert len(data["forecast"]) == 24
    assert "selected" in proc.stdout


def test_cli_invalid_horizon_exit_2(tmp_path: Path):
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": []}))
    (tmp_path / "items_config.json").write_text(json.dumps({"items": []}))
    proc = subprocess.run([sys.executable, "-m", "src.forecast.cli", "--item", "X", "--season", "s",
                           "--horizon", "5", "--csv-dir", str(tmp_path),
                           "--seasons", str(tmp_path / "seasons.json")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2
