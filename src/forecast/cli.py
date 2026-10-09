"""Command-line forecast runner over CSV exports.

    python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72 --out runs/iron.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .pipeline import (ForecastRequest, InsufficientDataError, InvalidRequestError, ListingLoader,
                       Season, run_forecast)


def _ts(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    t = pd.Timestamp(value)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def load_seasons(path: Path) -> list[Season]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Season(s["id"], s["name"], _ts(s["start"]), _ts(s.get("end"))) for s in raw.get("seasons", [])]


def csv_loader(csv_dir: Path, items_config: Path) -> ListingLoader:
    cfg = json.loads(Path(items_config).read_text(encoding="utf-8"))
    files = {it["name"]: Path(csv_dir) / it["csv"] for it in cfg.get("items", [])}
    cache: dict[str, pd.DataFrame] = {}

    def load(item: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
        if item not in files:
            raise InvalidRequestError(f"unknown item {item!r}; known: {sorted(files)}")
        if item not in cache:
            cache[item] = pd.read_csv(files[item])
        df = cache[item]
        ts = pd.to_datetime(df["created_at"], utc=True)
        mask = ts >= start
        if end is not None:
            mask &= ts < end
        return df[mask]

    return load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hourly price forecast with rolling-origin backtest")
    p.add_argument("--item", required=True)
    p.add_argument("--season", required=True, help="season id from seasons.json")
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--models", default="", help="comma-separated subset, e.g. ridge,hgb")
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--csv-dir", default=".")
    p.add_argument("--seasons", default="seasons.json")
    p.add_argument("--out", default=None, help="write full JSON result here")
    args = p.parse_args(argv)

    csv_dir = Path(args.csv_dir)
    req = ForecastRequest(args.item, args.season, args.horizon,
                          tuple(m for m in args.models.split(",") if m), args.coverage)
    try:
        seasons = load_seasons(Path(args.seasons))
        loader = csv_loader(csv_dir, csv_dir / "items_config.json")
        out = run_forecast(req, loader, seasons, source="csv", log=lambda s: print(f"  {s}", file=sys.stderr))
    except (InvalidRequestError, InsufficientDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"selected: {out['selected']['name']}  beats baseline: {out['beats_baseline']}  "
          f"skill vs naive: {out['skill_vs_naive']:+.1%}")
    print("model              mae_mean   mase   dir_acc")
    for m in out["backtest"]["models"]:
        print(f"{m['name']:<18} {m['mae_mean']:>8.3f} {m['mase_mean']:>6.2f} {m['direction_acc']:>8.2f}")
    last = min(23, len(out["forecast"]) - 1)
    print(f"next 24h: {out['forecast'][0]['pred']:.2f} -> {out['forecast'][last]['pred']:.2f}")
    for w in out["warnings"]:
        print(f"warning: {w}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
