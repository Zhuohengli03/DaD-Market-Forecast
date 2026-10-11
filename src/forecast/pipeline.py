"""run_forecast: listings -> hourly series -> backtest -> selection -> forecast + intervals."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .aggregate import block_series, hourly_series, prepare_target
from .backtest import aggregate_backtest, default_origin_step, independent_windows, run_backtest, summarize
from .intervals import apply_relative_bounds, empirical_coverage, quantile_bounds, relative_residuals
from .models import BASELINE_MODELS, MODEL_FACTORIES, make_model
from .reference import reference_curves
from .select import select_honestly

MIN_HOURS = 72
HORIZON_RANGE = (24, 168)
COVERAGE_RANGE = (0.5, 0.99)
TAIL_HOURS = 168
STALE_HOURS = 24
PRICE_FLOOR = 0.0
# forecast a steadier price: the average over 4-hour or daily blocks (models still learn from hours)
RESOLUTION_HOURS = {"1h": 1, "4h": 4, "1d": 24}

ListingLoader = Callable[[str, pd.Timestamp, "pd.Timestamp | None"], pd.DataFrame]
Logger = Callable[[str], None]


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "item"


class InsufficientDataError(ValueError):
    pass


class InvalidRequestError(ValueError):
    pass


@dataclass(frozen=True)
class Season:
    id: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    market_start: pd.Timestamp | None = None


@dataclass(frozen=True)
class ForecastRequest:
    item: str
    season_id: str
    horizon_hours: int = 72
    models: tuple[str, ...] = ()
    coverage: float = 0.9
    backtest_origin_step: int | None = None   # None: default_origin_step(horizon)
    resolution: str = "1h"


def _validate(req: ForecastRequest, seasons: list[Season]) -> tuple[Season, tuple[str, ...]]:
    lo, hi = HORIZON_RANGE
    if not lo <= req.horizon_hours <= hi:
        raise InvalidRequestError(f"预测时长必须在 {lo} 到 {hi} 小时之间，当前为 {req.horizon_hours}")
    clo, chi = COVERAGE_RANGE
    if not clo <= req.coverage <= chi:
        raise InvalidRequestError(f"区间可信度必须在 {clo} 到 {chi} 之间，当前为 {req.coverage}")
    if req.backtest_origin_step is not None and req.backtest_origin_step < 1:
        raise InvalidRequestError("回测间隔至少为 1 小时")
    unknown = [m for m in req.models if m not in MODEL_FACTORIES]
    if unknown:
        raise InvalidRequestError(f"不认识的方法 {unknown}，可选：{sorted(MODEL_FACTORIES)}")
    k = RESOLUTION_HOURS.get(req.resolution)
    if k is None:
        raise InvalidRequestError(f"时间粒度只能是 {sorted(RESOLUTION_HOURS)}，当前为 {req.resolution!r}")
    if req.horizon_hours % k:
        raise InvalidRequestError(f"预测时长 {req.horizon_hours} 小时不是 {k} 小时的整数倍")
    season = next((s for s in seasons if s.id == req.season_id), None)
    if season is None:
        raise InvalidRequestError(f"找不到赛季 {req.season_id!r}")
    chosen = tuple(req.models) if req.models else tuple(MODEL_FACTORIES)
    names = tuple(dict.fromkeys(BASELINE_MODELS + chosen))  # baselines first, no duplicates
    return season, names


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def to_blocks(hourly: np.ndarray, k: int) -> np.ndarray:
    """Hourly forecast -> the average of each k-hour block."""
    return hourly if k == 1 else hourly[: len(hourly) // k * k].reshape(-1, k).mean(axis=1)


def _fit_selected(selection, series: pd.DataFrame, season_start: pd.Timestamp, horizon: int, k: int = 1):
    """Refit members on the full series; return (point forecast in k-hour blocks, importance or None).

    At block resolution the baseline is the average of the last k hours, as it was in the backtest.
    """
    preds, importance = [], None
    for name, weight in selection.members:
        if name == "naive" and k > 1:
            y = series["median_price"].to_numpy(dtype=float)
            preds.append(weight * np.full(horizon // k, np.nanmean(y[-k:])))
            continue
        model = make_model(name).fit(series, season_start)
        preds.append(weight * to_blocks(model.predict(horizon), k))
        if importance is None:
            importance = model.importance()
    return np.sum(preds, axis=0), importance


def _naive_selection(result, skill: float, reason: str):
    from .select import Selection
    return Selection("naive", (("naive", 1.0),), False, skill, result.predictions["naive"], reason)


def run_forecast(request: ForecastRequest, load_listings: ListingLoader, seasons: list[Season],
                 source: str = "unknown", log: Logger | None = None, progress=None) -> dict:
    say = log or (lambda _s: None)
    step = progress or (lambda *a, **k: None)
    season, model_names = _validate(request, seasons)
    step("load")
    say(f"读取 {request.item} 在 {season.name} 的挂单")
    listings = load_listings(request.item, season.market_start if season.market_start is not None else season.start, season.end)
    full_series = hourly_series(listings)
    series, warnings = prepare_target(full_series)
    filled = int((~series["observed"]).sum()) if "observed" in series.columns else 0
    if filled:
        warnings.append(f"有 {filled} 个小时没有挂单，已用前一小时价格补齐（只用于建模，不计入评分）")
    if len(series) < MIN_HOURS:
        raise InsufficientDataError(
            f"至少需要 {MIN_HOURS} 个连续小时的数据，目前只有 {len(series)} 小时"
            f"（{season.name}，共 {len(listings)} 条挂单）。请先多采集几天。")
    say(f"共 {len(series)} 个小时的数据，开始回测 {len(model_names)} 种方法")

    step("backtest", 0, len(model_names))
    origin_step = request.backtest_origin_step or default_origin_step(request.horizon_hours)
    result = run_backtest(series, season.start, model_names, request.horizon_hours,
                          origin_step=origin_step, log=say,
                          on_model=lambda i, n, name: step("backtest", i, n, model=name))
    k = RESOLUTION_HOURS[request.resolution]
    result = aggregate_backtest(result, series["median_price"].to_numpy(dtype=float), k)
    step("select")
    warnings += list(result.warnings)
    warnings += [f"方法 {n} 运行失败：{err}" for n, err in result.failures.items()]
    metrics = summarize(result)
    # picked on the earlier backtests, judged (and the band built) on the later ones it was not picked on
    honest = select_honestly(result)
    selection, judged = honest.selection, honest.evaluation
    if honest.in_sample:
        warnings.append("回测次数不够分成两半，挑方法和检验用的是同一批回测，成绩会偏乐观")
    say(f"选用 {selection.name}（相对基准 {selection.skill_vs_naive:+.1%}）")

    if selection.reason:
        warnings.append(f"显示基准预测：{selection.reason}")
    step("fit", model=selection.name)
    try:
        point, importance = _fit_selected(selection, series, season.start, request.horizon_hours, k)
    except Exception as exc:
        warnings.append(f"{selection.name} 在完整赛季数据上运行失败（{type(exc).__name__}: {exc}），改用基准预测")
        selection = _naive_selection(judged, 0.0, "选中的方法在完整数据上运行失败")
        point, importance = _fit_selected(selection, series, season.start, request.horizon_hours, k)
    # errors as a share of the price, so a band learned when the price was high still fits today's price
    resid = relative_residuals(judged.actuals, selection.predictions, judged.last_values)
    lo, hi = quantile_bounds(resid, request.coverage)
    point = np.maximum(point, PRICE_FLOOR)
    last_level = float(np.nanmean(series["median_price"].to_numpy(dtype=float)[-k:]))
    lower, upper = apply_relative_bounds(point, lo, hi, last_level, floor=PRICE_FLOOR)
    coverage = empirical_coverage(resid, request.coverage)
    windows = independent_windows(result.origins, result.horizon * k)
    if windows < len(result.origins) and windows < 5:
        warnings.append(f"{len(result.origins)} 次回测的时间段互相重叠，相当于只有 {windows} 段独立数据，结论的把握比次数看起来要小")
    if request.horizon_hours > result.horizon * k:
        warnings.append(f"超过 {result.horizon * k} 小时的部分，区间宽度沿用回测到的最后一个值")

    last_ts = series.index[-1]
    current_offset = int((last_ts - season.start) / pd.Timedelta(hours=1))
    others = []
    for s in seasons:
        if s.id == season.id:
            continue
        other = hourly_series(load_listings(request.item, s.market_start if s.market_start is not None else s.start, s.end))
        if not other.empty:
            others.append((s.id, s.name, other, s.start))
    references = reference_curves(current_offset, request.horizon_hours, others)

    step("references")
    tail = block_series(full_series.iloc[-TAIL_HOURS:], k, last_ts + pd.Timedelta(hours=1))
    now = pd.Timestamp.now(tz="UTC")
    data_age_hours = float((now - last_ts) / pd.Timedelta(hours=1))
    season_ended = season.end is not None and season.end <= now
    if data_age_hours > STALE_HOURS and not season_ended:   # a finished season has no newer listings to collect
        warnings.append(f"最新的挂单是 {data_age_hours:.0f} 小时前的，预测从那时起算，不是从现在。请先采集新数据。")
    run_id = f"{now:%Y%m%dT%H%M%S}-{_slug(request.item)}-{uuid.uuid4().hex[:6]}"
    step("done", beats_baseline=selection.beats_baseline, selected=selection.name)
    return {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "request": {**asdict(request), "models": list(model_names)},
        "source": source,
        "season": {"id": season.id, "name": season.name, "start": season.start.isoformat(),
                   "end": season.end.isoformat() if season.end is not None else None, "ended": season_ended},
        "series_tail": [
            {"ts": ts.isoformat(), "median_price": _num(r.median_price), "p25": _num(r.p25),
             "p75": _num(r.p75), "volume": int(r.volume), "listings": int(r.listings)}
            for ts, r in tail.iterrows()],
        # each entry is one k-hour block: ts is where it starts, h how many hours ahead it ends
        "forecast": [
            {"ts": (last_ts + pd.Timedelta(hours=i * k + 1)).isoformat(), "h": (i + 1) * k,
             "pred": float(point[i]), "lower": float(lower[i]), "upper": float(upper[i])}
            for i in range(len(point))],
        "selected": {"name": selection.name,
                     "members": [{"name": n, "weight": w} for n, w in selection.members]},
        "backtest": {
            "origins": len(result.origins), "horizon": result.horizon * k, "resolution": request.resolution,
            "origin_step": origin_step, "independent_windows": windows,
            "in_sample": honest.in_sample, "selection_origins": honest.n_select, "evaluation_origins": honest.n_eval,
            "coverage": _num(coverage), "nominal_coverage": request.coverage,
            "models": [{"name": m.name, "mae_h1": _num(m.mae_h1), "mae_h24": _num(m.mae_h24),
                        "mae_mean": _num(m.mae_mean), "rmse_mean": _num(m.rmse_mean),
                        "mase_mean": _num(m.mase_mean), "direction_acc": _num(m.direction_acc),
                        "mdape_mean": _num(m.mdape_mean)}
                       for m in metrics],
            "per_horizon": {m.name: [_num(v) for v in m.per_horizon_mae] for m in metrics},
            "failures": result.failures,
        },
        "data_age_hours": data_age_hours,
        "baseline_reason": selection.reason or None,
        "beats_baseline": selection.beats_baseline,
        "skill_vs_naive": float(selection.skill_vs_naive),
        "feature_importance": [{"name": n, "value": float(v)} for n, v in importance] if importance else [],
        "references": [{"season_id": c.season_id, "name": c.name,
                        "points": [{"h": h, "median_price": p} for h, p in c.points]} for c in references],
        "warnings": warnings,
    }
