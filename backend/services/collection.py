from __future__ import annotations
import logging

from pathlib import Path

from ..settings import Settings
from .jobs import Job
from src.season_calendar import market_start


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


def market_not_open(state, season_id: str | None = None) -> str | None:
    """The trading-open date when the chosen season's market has not opened yet (nothing to collect), else None."""
    import pandas as pd
    season = pick_season(state, season_id)
    if season is None:
        return None
    opens = pd.Timestamp(market_start(season))
    return season.get("market_open") or opens.strftime("%Y-%m-%d") if opens > pd.Timestamp.now(tz="UTC") else None


def season_window(state, season_id: str | None = None) -> tuple[str | None, str | None]:
    """The chosen season's trading window: from the market opening (or season start) to the season end."""
    season = pick_season(state, season_id)
    return (market_start(season), season.get("end")) if season else (None, None)


def coverage_mark(state, name: str, season_id: str) -> str | None:
    """How far an ended season was collected for one item (written by the collector's --coverage-key)."""
    import json
    from src.api.storage import item_slug
    path = state.settings.data_dir / "coverage" / f"{item_slug(name)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(season_id)
    except (OSError, ValueError, AttributeError):
        return None


def season_progress(state, name: str, season_id: str | None = None) -> dict:
    """An ended season's collection progress for one item: collected through, complete, share of the window."""
    import pandas as pd
    season = pick_season(state, season_id)
    if season is None or not season.get("end"):
        return {"through": None, "complete": False, "share": None}
    start, end = pd.Timestamp(market_start(season)), pd.Timestamp(season["end"])
    through = coverage_mark(state, name, season["id"])
    if not through:
        return {"through": None, "complete": False, "share": 0.0}
    done = pd.Timestamp(through)
    share = 1.0 if done >= end else max(0.0, (done - start) / (end - start)) if end > start else 1.0
    return {"through": through, "complete": done >= end, "share": round(float(share), 4)}


def coverage_args(argv: list[str], season: dict | None) -> list[str]:
    """Ask the built-in collector to remember how far an ended season got, so a long backfill resumes there."""
    if season and season.get("end") and "src.api.collect" in argv:
        return argv + ["--coverage-key", season["id"]]
    return argv


def collect_range(state, name: str, season_id: str | None = None) -> tuple[str | None, str | None]:
    """Where a collection of one item starts and stops, season by season.

    An ended season is walked oldest-first from its first trading day, resuming where the last run stopped
    (its coverage mark). The current season starts at the newest listing already stored in it, or its first day.
    """
    import pandas as pd
    from ..data.repository import ListingQuery
    season = pick_season(state, season_id)
    if season is None:
        return None, None
    start, end = market_start(season), season.get("end")
    if end:
        return coverage_mark(state, name, season["id"]) or start, end
    try:
        rows, _ = state.repo.fetch(name, ListingQuery(start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None, page_size=1))
    except KeyError:
        rows = []
    return (rows[0]["created_at"] if rows else start), end


def start_collection(state, item: str, season_id: str | None = None) -> Job:
    from .cost import collect_estimate
    entry = state.items.get(item)  # KeyError if not tracked
    opens = market_not_open(state, season_id)
    if opens:
        raise RuntimeError(f"这个赛季的交易行 {opens} 才开放，之前没有交易数据可采集 / the market opens on {opens}")
    if season_progress(state, item, season_id)["complete"]:
        raise RuntimeError(f"{item} 这个赛季已经采集完整 / this season is already fully collected for {item}")
    since, until = collect_range(state, item, season_id)
    argv = coverage_args(state.collector_argv(entry, since, until), pick_season(state, season_id))
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
