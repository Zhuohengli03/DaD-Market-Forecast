from __future__ import annotations

import json

from ..data.repository import validate_item
from .jobs import Job


def start_collection(state, item: str) -> Job:
    item = validate_item(item)
    cfg_path = state.settings.items_config_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"items": []}
    entry = next((it for it in cfg.get("items", []) if it["name"] == item), None)
    if entry is None:
        raise KeyError(f"item {item!r} is not configured")
    script = state.settings.project_root / "src" / "api" / entry["file"]
    if not script.exists():
        raise FileNotFoundError(f"collector script missing: {script}")
    if state.jobs.find_active("collect", item=item):
        raise RuntimeError(f"collection for {item!r} is already running")
    job = state.jobs.create("collect", f"collect {item}", meta={"item": item})
    state.jobs.run_process(job, [state.settings.python_executable, str(script)], state.settings.project_root)
    return job
