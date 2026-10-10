import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_init_db_creates_the_builtin_database(tmp_path):
    (tmp_path / "items_config.json").write_text(json.dumps({"items": [{"name": "Iron Ore", "enabled": True}]}))
    out = subprocess.run([sys.executable, "-m", "backend.init_db", "--project-root", str(tmp_path)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "data" / "market.db").exists() and "market.db" in out.stdout
