from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from src.forecast.pipeline import Season
from src.season_calendar import market_start, with_market_opening
from ..schemas import SeasonIn
from ..services.envfile import atomic_write


class SeasonError(ValueError):
    pass


def _ts(value: str | None) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        t = pd.Timestamp(value)
    except Exception as exc:
        raise SeasonError(f"invalid timestamp {value!r}: {exc}") from None
    if pd.isna(t):
        raise SeasonError(f"invalid timestamp {value!r}")
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


_FAR_FUTURE = pd.Timestamp("2200-01-01T00:00:00Z")


class SeasonStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [with_market_opening(s) for s in json.loads(self._path.read_text(encoding="utf-8")).get("seasons", [])]

    def _write(self, seasons: list[dict]) -> None:
        atomic_write(self._path, json.dumps({"seasons": seasons}, ensure_ascii=False, indent=2))

    def list(self) -> list[dict]:
        return sorted(self._read(), key=lambda s: s["start"])

    def get(self, season_id: str) -> dict:
        for s in self._read():
            if s["id"] == season_id:
                return s
        raise KeyError(season_id)

    def _validate(self, data: SeasonIn, others: list[dict]) -> dict:
        start, end = _ts(data.start), _ts(data.end)
        if start is None:
            raise SeasonError("start is required")
        if end is not None and end <= start:
            raise SeasonError("end must be after start")
        opening = _ts(data.market_open)
        if opening is not None and (opening.date() < start.date() or (end is not None and opening >= end)):
            raise SeasonError("market opening must be within the season")
        for o in others:
            o_start, o_end = _ts(o["start"]), _ts(o["end"])
            latest_start = max(start, o_start)
            earliest_end = min(end or _FAR_FUTURE, o_end or _FAR_FUTURE)
            if latest_start < earliest_end:
                raise SeasonError(f"range overlaps season {o['name']!r}")
        return {"name": data.name, "start": start.isoformat(),
                "end": end.isoformat() if end is not None else None, "note": data.note,
                "market_open": data.market_open}

    def add(self, data: SeasonIn) -> dict:
        seasons = self._read()
        record = {"id": f"s-{uuid.uuid4().hex[:8]}", **self._validate(data, seasons)}
        self._write(seasons + [record])
        return record

    def update(self, season_id: str, data: SeasonIn) -> dict:
        seasons = self._read()
        if not any(s["id"] == season_id for s in seasons):
            raise KeyError(season_id)
        others = [s for s in seasons if s["id"] != season_id]
        if "market_open" not in data.model_fields_set:
            data = data.model_copy(update={"market_open": next(s for s in seasons if s["id"] == season_id).get("market_open")})
        record = {"id": season_id, **self._validate(data, others)}
        self._write([record if s["id"] == season_id else s for s in seasons])
        return record

    def delete(self, season_id: str) -> None:
        seasons = self._read()
        if not any(s["id"] == season_id for s in seasons):
            raise KeyError(season_id)
        self._write([s for s in seasons if s["id"] != season_id])

    def import_records(self, records: list[dict], replace: bool = False) -> tuple[int, int]:
        """Add seasons with given ids. replace=True swaps the whole list; otherwise overlapping ones are skipped."""
        previous = {s["id"]: s for s in self._read()}
        current = [] if replace else list(previous.values())
        added = skipped = 0
        for r in records:
            if any(s["id"] == r["id"] for s in current):
                skipped += 1
                continue
            try:
                r = with_market_opening(r)
                # Patch dates don't carry trading dates. Keep manually corrected openings on reimport.
                old = previous.get(r["id"], {})
                if "market_open" in old:
                    r["market_open"] = old["market_open"]
                clean = self._validate(SeasonIn(name=r["name"], start=r["start"], end=r["end"], note=r.get("note", ""), market_open=r.get("market_open")), current)
            except SeasonError:
                skipped += 1
                continue
            current.append({"id": r["id"], **clean})
            added += 1
        self._write(current)
        return added, skipped

    def to_pipeline(self) -> list[Season]:
        return [Season(s["id"], s["name"], _ts(s["start"]), _ts(s["end"]), _ts(market_start(s))) for s in self.list()]

    def assign(self, ts: pd.Timestamp) -> str | None:
        for s in self.to_pipeline():
            if ts >= s.start and (s.end is None or ts < s.end):
                return s.id
        return None
