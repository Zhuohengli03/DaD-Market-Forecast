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


def _interval_hours(settings_path: Path | None) -> int | None:
    if settings_path and Path(settings_path).exists():
        return json.loads(Path(settings_path).read_text(encoding="utf-8")).get("collect_interval_hours")
    return None


def tasks_from_items(items_path: Path, project_root: Path = PROJECT_ROOT, settings_path: Path | None = None) -> list[dict]:
    items_path = Path(items_path)
    if not items_path.exists():
        items_path = Path(project_root) / "items_config.json"
    if not items_path.exists():
        return []
    items = json.loads(items_path.read_text(encoding="utf-8")).get("items", [])
    hours = _interval_hours(settings_path)
    tasks = []
    for it in items:
        if not it.get("enabled", True):
            continue
        args = ["--item", it["name"]] + (["--item-id", it["item_id"]] if it.get("item_id") else [])
        tasks.append({
            "name": f"Collect {it['name']}",
            "item": it["name"],
            "script_path": str(Path(project_root) / "src" / "api" / "collect.py"),
            "args": args,
            "schedule_type": "interval" if hours else "daily",
            "schedule_value": int(hours) * 60 if hours else it.get("schedule", DEFAULT_TIME),
            "enabled": True,
            "description": f"每 {hours} 小时采集 {it['name']}" if hours else f"每天 {it.get('schedule', DEFAULT_TIME)} 采集 {it['name']}",
        })
    return tasks


TASKS = tasks_from_items(Path(os.getenv("DARKER_ITEMS_CONFIG", PROJECT_ROOT / "data" / "items.json")),
                         settings_path=Path(os.getenv("DARKER_SETTINGS", PROJECT_ROOT / "data" / "settings.json")))

SCHEDULER_CONFIG = {
    "log_file": "scheduler.log",
    "log_level": "INFO",
    "check_interval": 1,
    "max_concurrent_tasks": 1,  # one DarkerDB key allows 60 requests/min; collectors run one at a time
}
