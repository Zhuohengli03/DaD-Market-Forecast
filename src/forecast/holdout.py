"""Season holdout: how well would the forecaster have done on a season it had not seen the end of?

The method is chosen (backtest + selection) on the first `train_frac` of the season only and then frozen.
From the cut onward it forecasts `horizon` hours ahead once every `step` hours, refitting the chosen method on
everything known up to that point, and is scored against what really happened and against repeating the last hour.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .aggregate import hourly_series, prepare_target
from .backtest import aggregate_backtest, default_origin_step, run_backtest, summarize
from .intervals import apply_relative_bounds, quantile_bounds, relative_residuals
from .models import BASELINE_MODELS, MODEL_FACTORIES
from .pipeline import MIN_HOURS, PRICE_FLOOR, RESOLUTION_HOURS, InsufficientDataError, InvalidRequestError, ListingLoader, \
    Season, _fit_selected, _num, _slug
from .select import select_honestly

TRAIN_FRAC_RANGE = (0.5, 0.9)


@dataclass(frozen=True)
class HoldoutRequest:
    item: str
    season_id: str
    train_frac: float = 0.7
    horizon_hours: int = 24
    models: tuple[str, ...] = ()
    coverage: float = 0.9
    resolution: str = "1h"


def _direction_acc(actual: np.ndarray, pred: np.ndarray, last: np.ndarray) -> float | None:
    """Share of moves called right; a step counts only when both the price and the forecast moved."""
    a, p = np.sign(actual - last), np.sign(pred - last)
    scored = ~np.isnan(actual) & (a != 0) & (p != 0)
    return float(np.mean(a[scored] == p[scored])) if scored.any() else None


def _score(actual, pred, naive, lower, upper, last) -> dict:
    ok = ~np.isnan(actual)
    a, p, nv = actual[ok], pred[ok], naive[ok]
    mae, naive_mae = float(np.mean(np.abs(a - p))), float(np.mean(np.abs(a - nv)))
    pos = a > 0
    return {
        "mae": mae, "naive_mae": naive_mae,
        "skill": 1.0 - mae / naive_mae if naive_mae > 0 else 0.0,
        "mape": float(np.mean(np.abs(a[pos] - p[pos]) / a[pos])) if pos.any() else None,
        # typical error: half of the forecasts were closer than this; a rare spike barely moves it
        "mdape": float(np.median(np.abs(a[pos] - p[pos]) / a[pos])) if pos.any() else None,
        "direction_acc": _direction_acc(actual, pred, last),
        "coverage": float(np.mean((a >= lower[ok]) & (a <= upper[ok]))),
        "scored_hours": int(ok.sum()),
    }


def _block_mean(hourly: np.ndarray, k: int) -> np.ndarray:
    """Mean of each k-hour block of real prices; a block with no real listing stays unscored (NaN)."""
    blocks = hourly[: len(hourly) // k * k].reshape(-1, k)
    seen = (~np.isnan(blocks)).sum(axis=1)
    return np.where(seen > 0, np.nansum(blocks, axis=1) / np.maximum(seen, 1), np.nan)


def run_holdout(series: pd.DataFrame, season_start: pd.Timestamp, model_names: tuple[str, ...],
                train_frac: float = 0.7, horizon: int = 24, coverage: float = 0.9, origin_step: int | None = None,
                log=None, progress=None, resolution: str = "1h") -> dict:
    k = RESOLUTION_HOURS[resolution]
    say, step = log or (lambda _s: None), progress or (lambda *a, **k: None)
    n = len(series)
    split = int(n * train_frac)
    if split < MIN_HOURS or n - split < horizon:
        raise InsufficientDataError(
            f"这段数据只有 {n} 小时：前 {train_frac:.0%} 至少要 {MIN_HOURS} 小时用来选方法，"
            f"后面至少要 {horizon} 小时用来检验。请先把这个赛季采集完整。")
    train = series.iloc[:split]
    say(f"前 {split} 小时用来选方法，后 {n - split} 小时用来检验")
    step("backtest", 0, len(model_names))
    result = run_backtest(train, season_start, model_names, horizon, origin_step=origin_step or default_origin_step(horizon), log=say,
                          on_model=lambda i, k, name: step("backtest", i, k, model=name))
    y = series["median_price"].to_numpy(dtype=float)
    result = aggregate_backtest(result, y[:split], k)
    honest = select_honestly(result)          # within the training part too: pick early, judge later
    selection, judged = honest.selection, honest.evaluation
    lo, hi = quantile_bounds(relative_residuals(judged.actuals, selection.predictions, judged.last_values), coverage)
    say(f"选定 {selection.name}，之后不再更换")

    observed = series["observed"].to_numpy(dtype=bool) if "observed" in series.columns else np.ones(n, bool)
    origins = list(range(split, n - horizon + 1, horizon))
    rows, windows = [], []
    for i, o in enumerate(origins, start=1):
        step("holdout", i, len(origins))
        pred, _ = _fit_selected(selection, series.iloc[:o], season_start, horizon, k)
        pred = np.maximum(pred, PRICE_FLOOR)
        level = float(np.nanmean(y[o - k:o]))          # the last hour, or the last block's average
        lower, upper = apply_relative_bounds(pred, lo, hi, level, floor=PRICE_FLOOR)
        hourly = np.where(observed[o:o + horizon], y[o:o + horizon], np.nan)   # filled hours are not scored
        actual = hourly if k == 1 else _block_mean(hourly, k)
        naive = np.full(len(pred), level)
        last = np.full(len(pred), level)
        rows.append((actual, pred, naive, lower, upper, last))
        windows.append({"origin": series.index[o].isoformat(), **_score(actual, pred, naive, lower, upper, last)})
    cols = [np.concatenate(c) for c in zip(*rows)]
    index = [ts for o in origins for ts in series.index[o:o + horizon:k]]   # each point starts one block
    return {
        "split_ts": series.index[split].isoformat(),
        "train_hours": split, "test_hours": n - split,
        "horizon": horizon, "resolution": resolution, "nominal_coverage": coverage,
        "selected": {"name": selection.name, "members": [{"name": m, "weight": w} for m, w in selection.members]},
        "train_skill": float(selection.skill_vs_naive),
        "metrics": _score(*cols),
        "windows": windows,
        "points": [{"ts": ts.isoformat(), "actual": _num(a), "pred": float(p), "naive": float(nv),
                    "lower": float(lw), "upper": float(up)}
                   for ts, a, p, nv, lw, up in zip(index, cols[0], cols[1], cols[2], cols[3], cols[4])],
        "train_tail": [{"ts": ts.isoformat(), "actual": _num(v)}
                       for ts, v in series["median_price"].iloc[max(0, split - 7 * 24):split].items()],
        "warnings": list(result.warnings) + ([f"选方法时：{selection.reason}"] if selection.reason else []),
    }


def run_season_holdout(request: HoldoutRequest, load_listings: ListingLoader, seasons: list[Season],
                       log=None, progress=None) -> dict:
    lo, hi = TRAIN_FRAC_RANGE
    if not lo <= request.train_frac <= hi:
        raise InvalidRequestError(f"训练比例必须在 {lo:.0%} 到 {hi:.0%} 之间")
    season = next((s for s in seasons if s.id == request.season_id), None)
    if season is None:
        raise InvalidRequestError(f"找不到赛季 {request.season_id!r}")
    unknown = [m for m in request.models if m not in MODEL_FACTORIES]
    if unknown:
        raise InvalidRequestError(f"不认识的方法 {unknown}")
    names = tuple(dict.fromkeys(BASELINE_MODELS + (tuple(request.models) or tuple(MODEL_FACTORIES))))
    (progress or (lambda *a, **k: None))("load")
    start = season.market_start if season.market_start is not None else season.start
    listings = load_listings(request.item, start, season.end)
    series, warnings = prepare_target(hourly_series(listings))
    if request.resolution not in RESOLUTION_HOURS:
        raise InvalidRequestError(f"时间粒度只能是 {sorted(RESOLUTION_HOURS)}")
    if request.horizon_hours % RESOLUTION_HOURS[request.resolution]:
        raise InvalidRequestError("预测时长必须是时间粒度的整数倍")
    out = run_holdout(series, season.start, names, request.train_frac, request.horizon_hours, request.coverage,
                      log=log, progress=progress, resolution=request.resolution)
    now = pd.Timestamp.now(tz="UTC")
    return {
        "run_id": f"{now:%Y%m%dT%H%M%S}-holdout-{_slug(request.item)}-{uuid.uuid4().hex[:6]}",
        "created_at": now.isoformat(),
        "item": request.item,
        "request": {**asdict(request), "models": list(names)},
        "season": {"id": season.id, "name": season.name, "start": season.start.isoformat(),
                   "end": season.end.isoformat() if season.end is not None else None},
        "listings": len(listings),
        **out,
        "warnings": warnings + out["warnings"],
    }
