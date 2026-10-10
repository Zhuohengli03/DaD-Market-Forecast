"""App preferences chosen in the setup wizard (data/settings.json)."""
from __future__ import annotations

import json
from pathlib import Path

from ..services.envfile import atomic_write

DEFAULTS = {"initialized_at": None, "collect_interval_hours": 6, "auto_collect": False}
INTERVAL_CHOICES = (1, 2, 3, 6, 12, 24)


class AppSettings:
    def __init__(self, path: Path):
        self._path = Path(path)

    def get(self) -> dict:
        data = dict(DEFAULTS)
        if self._path.exists():
            data.update({k: v for k, v in json.loads(self._path.read_text(encoding="utf-8")).items() if k in DEFAULTS})
        return data

    def update(self, **changes) -> dict:
        data = {**self.get(), **{k: v for k, v in changes.items() if k in DEFAULTS}}
        atomic_write(self._path, json.dumps(data, ensure_ascii=False, indent=2))
        return data
