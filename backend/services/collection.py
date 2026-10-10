from __future__ import annotations
import logging

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


def collection_states(state) -> dict[str, str]:
    """Each tracked item that is being collected ("collecting") or queued for collection ("waiting").

    Sources: single-item collect jobs, the "collect all" and first-run jobs (per-item rows in their progress), and
    the scheduler process, which writes data/scheduler_state.json while it runs.
    """
    import json
    out: dict[str, str] = {}

    def put(name, s):
        if s in ("collecting", "waiting") and name and out.get(name) != "collecting":
            out[name] = s
    for job in state.jobs.active():
        if job.kind == "collect" and job.meta.get("item") not in (None, "*"):
            put(job.meta["item"], "collecting")
        for row in ((job.progress or {}).get("detail") or {}).get("items") or []:
            put(row.get("name"), row.get("status"))
    path = state.settings.data_dir / "scheduler_state.json"
    if path.exists() and state.scheduler.running():
        try:
            tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks") or {}
            items = {t["name"]: t.get("item") for t in state.scheduler.tasks()}
            for task, s in tasks.items():
                put(items.get(task), s)
        except (OSError, ValueError, KeyError):
            logging.getLogger(__name__).warning("unreadable scheduler state %s", path, exc_info=True)
    return out
