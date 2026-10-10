"""Tracked items (data/items.json): which DarkerDB items this install collects and forecasts."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..services.envfile import atomic_write


from src.api.storage import item_slug as slug  # noqa: E402,F401  one slug rule for collector and reader


class ItemsStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def read(self) -> dict:
        if not self._path.exists():
            return {"items": [], "categories": {}}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        data.setdefault("items", [])
        data.setdefault("categories", {})
        return data

    def write(self, config: dict) -> None:
        atomic_write(self._path, json.dumps(config, ensure_ascii=False, indent=2))

    def list(self, enabled_only: bool = False) -> list[dict]:
        items = self.read()["items"]
        return [i for i in items if i.get("enabled", True)] if enabled_only else items

    def get(self, name: str) -> dict:
        for it in self.list():
            if it["name"] == name:
                return it
        raise KeyError(f"item {name!r} is not tracked")

    def csv_name(self, item: dict) -> str:
        return item.get("csv") or f"{slug(item['name'])}.csv"

    def track(self, name: str, item_id: str, category: str = "", description: str = "") -> dict:
        cfg = self.read()
        if any(item_id and it.get("item_id") == item_id for it in cfg["items"]):
            raise ValueError(f"{name} 已经在追踪列表里 / already tracked")
        # a starter or legacy entry tracked by name only is the same item: give it the id instead of duplicating
        for it in cfg["items"]:
            if it["name"].lower() == name.lower() and not it.get("item_id"):
                it["item_id"] = item_id
                self.write(cfg)
                return it
        taken = {it["name"].lower() for it in cfg["items"]}
        if name.lower() in taken:  # another rarity of the same item: keep names unique (they name the data files)
            name = f"{name} ({description or item_id.rsplit('_', 1)[-1]})"
            if name.lower() in taken:
                raise ValueError(f"{name} 已经在追踪列表里 / already tracked")
        entry = {"name": name, "item_id": item_id, "category": category, "description": description, "enabled": True}
        cfg["items"].append(entry)
        self.write(cfg)
        return entry

    def untrack(self, name: str) -> None:
        cfg = self.read()
        kept = [it for it in cfg["items"] if it["name"] != name]
        if len(kept) == len(cfg["items"]):
            raise KeyError(f"item {name!r} is not tracked")
        cfg["items"] = kept
        self.write(cfg)
