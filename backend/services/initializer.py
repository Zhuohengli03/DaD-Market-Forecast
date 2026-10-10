"""First-run initialization: catalog, seasons, tracked items, first collection and a first forecast."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from src.forecast.pipeline import ForecastRequest, InsufficientDataError, run_forecast
from .catalog_sync import sync_catalog

QUICK_MODELS = ("ets", "hgb")  # keeps the first forecast to well under a minute per item


def _run_collector(jm, job, argv, cwd, env, on_progress) -> int:
    """Run one collector process, forwarding its @@progress lines; returns listings saved."""
    proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)
    saved, error = 0, None
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip("\n")
        if line.startswith("@@"):
            try:
                obj = json.loads(line[2:])
            except ValueError:
                continue
            if obj.get("stage") == "collect":
                on_progress(obj.get("done"), obj.get("new", 0))
            elif obj.get("stage") == "done":
                saved = int(obj.get("new", 0))
            elif obj.get("stage") == "error":
                error = obj.get("code", "api")
        else:
            jm.append_log(job, line)
    code = proc.wait()
    if error:
        raise RuntimeError(f"collector:{error}")
    if code != 0:
        raise RuntimeError(f"collector exit code {code}")
    return saved


def run_initialization(state, job, item_ids: list[str], max_pages: int) -> str:
    jm, settings = state.jobs, state.settings
    rows: list[dict] = []

    def publish(stage: str, step: int):
        jm.progress(job, stage, step, 4, items=[dict(r) for r in rows])

    # 1. catalog
    publish("catalog", 0)
    if not state.catalog.count():
        from ..api.catalog import client_for_state
        sync_catalog(client_for_state(state), state.catalog, lambda s: jm.append_log(job, s))
    # 2. seasons
    publish("seasons", 1)
    if not state.seasons.list():
        from ..api.seasons import import_records_from_darkerdb
        import_records_from_darkerdb(state)
    seasons = state.seasons.list()
    current = seasons[-1] if seasons else None
    # 3. tracked items
    for iid in item_ids:
        found = state.catalog.variant(iid)
        if found is None:
            continue
        raw, variants = found
        name = str(raw.get("name") or iid) + (f" ({raw.get('rarity')})" if len(variants) > 1 else "")
        try:
            state.items.track(name, iid, str(raw.get("item_type", "")))
        except ValueError:
            pass  # already tracked
        rows.append({"item_id": iid, "name": name, "name_zh": raw.get("name_zh"), "status": "waiting",
                     "pages": 0, "collected": 0, "forecast": None, "run_id": None, "hours": None})
    state.item_stats.invalidate()
    # 4. first collection of the current season, one item at a time (they share one API key's rate limit)
    from .collection import collector_env
    import os
    env = {**os.environ, **collector_env(settings), "PYTHONUNBUFFERED": "1"}
    for row in rows:
        row["status"] = "collecting"
        publish("collect", 2)
        entry = state.items.get(row["name"])
        argv = state.collector_argv(entry, current["start"] if current else None) + ["--max-pages", str(max_pages)]

        def on_page(page, new, row=row):
            row["pages"], row["collected"] = page or 0, new
            publish("collect", 2)
        try:
            row["collected"] = _run_collector(jm, job, argv, settings.project_root, env, on_page)
            row["status"] = "collected"
        except RuntimeError as exc:
            row["status"] = "collect_failed"
            row["error"] = "auth" if "auth" in str(exc) else "api"
        publish("collect", 2)
    state.item_stats.invalidate()
    # 5. first forecast for each item
    for row in rows:
        if row["status"] != "collected" or current is None:
            continue
        row["status"] = "forecasting"
        publish("forecast", 3)
        try:
            req = ForecastRequest(row["name"], current["id"], 24, QUICK_MODELS)
            result = run_forecast(req, state.repo.fetch_all, state.seasons.to_pipeline(), source=state.repo.source_name(),
                                  log=lambda s: jm.append_log(job, s))
            row["run_id"], row["forecast"] = state.runs.save(result), "ok"
        except InsufficientDataError as exc:
            row["forecast"] = "not_enough_data"
            digits = [int(x) for x in str(exc).split() if x.isdigit()]
            row["hours"] = digits[1] if len(digits) > 1 else None
        except Exception as exc:  # a forecast problem must not fail the whole setup
            row["forecast"] = "failed"
            jm.append_log(job, f"forecast {row['name']}: {type(exc).__name__}: {exc}")
        row["status"] = "done"
        publish("forecast", 3)
    state.app_settings.update(initialized_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    jm.progress(job, "done", 4, 4, items=[dict(r) for r in rows])
    return "init"


PARALLEL_COLLECTS = 3  # collectors share one key's per-minute limit; more workers would only queue on it


def run_collect_all(state, job) -> str:
    """Collect every enabled tracked item, up to PARALLEL_COLLECTS at a time, with per-item progress."""
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from .collection import collector_env
    jm, settings = state.jobs, state.settings
    rows = [{"item_id": it.get("item_id") or it["name"], "name": it["name"], "name_zh": None, "status": "waiting",
             "pages": 0, "collected": 0, "forecast": None, "run_id": None, "hours": None}
            for it in state.items.list(enabled_only=True)]
    names_zh = {v["id"]: v["name_zh"] for variants in state.catalog.types().values() for v in variants if v.get("name_zh")}
    for r in rows:
        r["name_zh"] = names_zh.get(r["item_id"])
    env = {**os.environ, **collector_env(settings), "PYTHONUNBUFFERED": "1"}
    lock = threading.Lock()

    def publish(stage):
        with lock:
            jm.progress(job, stage, sum(r["status"] in ("collected", "collect_failed") for r in rows), len(rows),
                        items=[dict(r) for r in rows])

    def collect(row):
        row["status"] = "collecting"
        publish("collect")

        def on_page(page, new):
            row["pages"], row["collected"] = page or 0, new
            publish("collect")
        try:
            row["collected"] = _run_collector(jm, job, state.collector_argv(state.items.get(row["name"]), None),
                                              settings.project_root, env, on_page)
            row["status"] = "collected"
        except RuntimeError as exc:
            row["status"], row["error"] = "collect_failed", ("auth" if "auth" in str(exc) else "api")
        publish("collect")
    with ThreadPoolExecutor(max_workers=PARALLEL_COLLECTS) as pool:
        list(pool.map(collect, rows))
    state.item_stats.invalidate()
    publish("done")
    return "collect-all"
