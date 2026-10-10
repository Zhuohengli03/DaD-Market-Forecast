"""App preferences chosen in the setup wizard (data/settings.json)."""
from __future__ import annotations

import json
from pathlib import Path

from ..services.envfile import atomic_write

DEFAULTS = {"initialized_at": None, "collect_interval_seconds": 6 * 3600, "auto_collect": False}
MIN_INTERVAL_SECONDS = 30
MAX_INTERVAL_SECONDS = 30 * 86400
DARKERDB_LIVE_RATE_PER_MINUTE = 60  # darkerdb.live scope default; plans may differ


class AppSettings:
    def __init__(self, path: Path):
        self._path = Path(path)

    def get(self) -> dict:
        data = dict(DEFAULTS)
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if "collect_interval_hours" in raw and "collect_interval_seconds" not in raw:  # older setting
                raw["collect_interval_seconds"] = int(raw["collect_interval_hours"]) * 3600
            data.update({k: v for k, v in raw.items() if k in DEFAULTS})
        return data

    def update(self, **changes) -> dict:
        data = {**self.get(), **{k: v for k, v in changes.items() if k in DEFAULTS}}
        atomic_write(self._path, json.dumps(data, ensure_ascii=False, indent=2))
        return data
