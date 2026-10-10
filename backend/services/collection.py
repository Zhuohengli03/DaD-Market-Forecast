from __future__ import annotations

from pathlib import Path

from ..settings import Settings
from .jobs import Job


def collector_argv(settings: Settings, entry: dict) -> list[str]:
    """Command that collects one tracked item. Legacy entries may point at a script inside src/api/."""
    if entry.get("item_id") or not entry.get("file"):
        argv = [settings.python_executable, "-m", "src.api.collect", "--item", entry["name"]]
        if entry.get("item_id"):
            argv += ["--item-id", entry["item_id"]]
        return argv
    api_dir = (settings.project_root / "src" / "api").resolve()
    script = (api_dir / entry["file"]).resolve()
    if api_dir not in script.parents:
        raise ValueError(f"collector script must live in src/api: {entry['file']}")
    if not script.exists():
        raise FileNotFoundError(f"collector script missing: {script}")
    return [settings.python_executable, str(script)]


def collector_env(settings: Settings) -> dict[str, str]:
    return {"DARKER_DATA_DIR": str(settings.data_dir), "DARKER_PROGRESS": "1"}


def start_collection(state, item: str) -> Job:
    entry = state.items.get(item)  # KeyError if not tracked
    argv = collector_argv(state.settings, entry)
    if state.jobs.find_active("collect", item=item):
        raise RuntimeError(f"{item} 正在采集中 / collection for {item!r} is already running")
    job = state.jobs.create("collect", f"采集 {item} / collect {item}", meta={"item": item})
    state.jobs.run_process(job, argv, state.settings.project_root, env=collector_env(state.settings))
    return job
