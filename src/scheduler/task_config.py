#!/usr/bin/env python3
"""Scheduled tasks: one collection run per enabled tracked item.

Tracked items live in data/items.json (created on first start from items_config.json). Each item may set
"schedule": "HH:MM" (daily, local time); the default is 23:00. Add or remove items in the web app's Catalog
or Settings page instead of editing this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIME = "23:00"


def _interval_seconds(settings_path: Path | None) -> int | None:
    if settings_path and Path(settings_path).exists():
        raw = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        if raw.get("collect_interval_seconds"):
            return int(raw["collect_interval_seconds"])
        if raw.get("collect_interval_hours"):
            return int(raw["collect_interval_hours"]) * 3600
    return None


def format_interval(seconds: int) -> str:
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = [(d, "天"), (h, "时"), (m, "分"), (s, "秒")]
    return "".join(f"{v}{u}" for v, u in parts if v) or "0秒"


def item_schedule(it: dict, default_seconds: int | None) -> tuple[str, object]:
    """(schedule_type, value) for one tracked item: its own schedule, else the global default."""
    sched = it.get("schedule")
    if isinstance(sched, str) and sched:
        return "daily_times", [sched]
    if isinstance(sched, dict):
        if sched.get("mode") == "interval" and sched.get("seconds"):
            return "seconds", int(sched["seconds"])
        if sched.get("mode") == "daily" and sched.get("times"):
            return "daily_times", sorted(set(sched["times"]))
    if default_seconds:
        return "seconds", int(default_seconds)
    return "daily_times", [DEFAULT_TIME]


def describe(kind: str, value) -> str:
    return f"每 {format_interval(value)}" if kind == "seconds" else "每天 " + "、".join(value)


def tasks_from_items(items_path: Path, project_root: Path = PROJECT_ROOT, settings_path: Path | None = None) -> list[dict]:
    items_path = Path(items_path)
    if not items_path.exists():
        items_path = Path(project_root) / "items_config.json"
    if not items_path.exists():
        return []
    items = json.loads(items_path.read_text(encoding="utf-8")).get("items", [])
    every = _interval_seconds(settings_path)
    tasks = []
    for it in items:
        if not it.get("enabled", True) or it.get("scheduled") is False:   # unticked on the schedule page
            continue
        kind, value = item_schedule(it, every)
        args = ["--item", it["name"]] + (["--item-id", it["item_id"]] if it.get("item_id") else [])
        tasks.append({
            "name": f"Collect {it['name']}",
            "item": it["name"],
            "script_path": str(Path(project_root) / "src" / "api" / "collect.py"),
            "args": args,
            "schedule_type": kind,
            "schedule_value": value,
            "enabled": True,
            "description": f"{describe(kind, value)} 采集 {it['name']}",
        })
    return tasks


TASKS = tasks_from_items(Path(os.getenv("DARKER_ITEMS_CONFIG", PROJECT_ROOT / "data" / "items.json")),
                         settings_path=Path(os.getenv("DARKER_SETTINGS", PROJECT_ROOT / "data" / "settings.json")))

SCHEDULER_CONFIG = {
    "log_file": "scheduler.log",
    "log_level": "INFO",
    "check_interval": 1,
    "max_concurrent_tasks": 3,  # collectors share one key's 60 requests/min; more would only queue on it
}
