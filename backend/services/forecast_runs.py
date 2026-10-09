from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.forecast.aggregate import hourly_series

SUMMARY_KEYS = ("run_id", "created_at", "source", "beats_baseline", "skill_vs_naive")


class RunStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not run_id or not run_id.replace("-", "").replace("_", "").isalnum():
            raise KeyError(run_id)
        return self._dir / f"{run_id}.json"

    def save(self, result: dict) -> str:
        self._path(result["run_id"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result["run_id"]

    def get(self, run_id: str) -> dict:
        p = self._path(run_id)
        if not p.exists():
            raise KeyError(run_id)
        return json.loads(p.read_text(encoding="utf-8"))

    def delete(self, run_id: str) -> None:
        p = self._path(run_id)
        if not p.exists():
            raise KeyError(run_id)
        p.unlink()

    @staticmethod
    def summary(run: dict) -> dict:
        return {**{k: run.get(k) for k in SUMMARY_KEYS},
                "item": run["request"]["item"], "season_id": run["season"]["id"],
                "season_name": run["season"]["name"], "horizon_hours": run["request"]["horizon_hours"],
                "selected": run["selected"]["name"], "warnings": len(run.get("warnings", []))}

    def list(self) -> list[dict]:
        runs = []
        for p in self._dir.glob("*.json"):
            try:
                runs.append(self.summary(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    def actuals(self, run: dict, repo) -> list[dict]:
        fc = run["forecast"]
        start = pd.Timestamp(fc[0]["ts"])
        end = pd.Timestamp(fc[-1]["ts"]) + pd.Timedelta(hours=1)
        df = repo.fetch_all(run["request"]["item"], start, end)
        if df.empty:
            return []
        series = hourly_series(df)
        wanted = {pd.Timestamp(f["ts"]) for f in fc}
        return [{"ts": ts.isoformat(), "median_price": float(r.median_price)}
                for ts, r in series.iterrows() if ts in wanted and not math.isnan(r.median_price)]
