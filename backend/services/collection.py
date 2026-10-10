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


def pick_season(state, season_id: str | None = None) -> dict | None:
    """The chosen season, or the current one (latest start) when none is given."""
    import pandas as pd
    seasons = [s for s in state.seasons.list() if s.get("start")]
    if season_id:
        found = next((s for s in seasons if s.get("id") == season_id), None)
        if found is None:
            raise KeyError(f"没有这个赛季 / unknown season {season_id}")
        return found
    return max(seasons, key=lambda s: pd.Timestamp(s["start"])) if seasons else None


def collect_range(state, name: str, season_id: str | None = None) -> tuple[str | None, str | None]:
    """Where a collection of one item starts and stops, season by season.

    Starts at the newest listing already stored inside the season (so nothing is fetched twice), or at the
    season's first day when none is stored; stops at the season's end (open-ended for the current season).
    """
    import pandas as pd
    from ..data.repository import ListingQuery
    season = pick_season(state, season_id)
    if season is None:
        return None, None
    start, end = season["start"], season.get("end")
    try:
        rows, _ = state.repo.fetch(name, ListingQuery(start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None, page_size=1))
    except KeyError:
        rows = []
    return (rows[0]["created_at"] if rows else start), end


def start_collection(state, item: str, season_id: str | None = None) -> Job:
    from .cost import collect_estimate
    entry = state.items.get(item)  # KeyError if not tracked
    since, until = collect_range(state, item, season_id)
    argv = state.collector_argv(entry, since, until)
    if state.jobs.find_active("collect", item=item):
        raise RuntimeError(f"{item} 正在采集中 / collection for {item!r} is already running")
    job = state.jobs.create("collect", f"采集 {item} / collect {item}", meta={"item": item})
    state.jobs.set_estimate(job, collect_estimate(state, entry, since=since, until=until)["requests"])
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
