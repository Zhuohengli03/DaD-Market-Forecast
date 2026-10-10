"""run_forecast: listings -> hourly series -> backtest -> selection -> forecast + intervals."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .aggregate import hourly_series, prepare_target
from .backtest import run_backtest, summarize
from .intervals import apply_bounds, empirical_coverage, quantile_bounds
from .models import BASELINE_MODELS, MODEL_FACTORIES, make_model
from .reference import reference_curves
from .select import select_model

MIN_HOURS = 72
HORIZON_RANGE = (24, 168)
COVERAGE_RANGE = (0.5, 0.99)
TAIL_HOURS = 168
STALE_HOURS = 24
PRICE_FLOOR = 0.0

ListingLoader = Callable[[str, pd.Timestamp, "pd.Timestamp | None"], pd.DataFrame]
Logger = Callable[[str], None]


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


@dataclass(frozen=True)
class ForecastRequest:
    item: str
    season_id: str
    horizon_hours: int = 72
    models: tuple[str, ...] = ()
    coverage: float = 0.9
    backtest_origin_step: int = 6


def _validate(req: ForecastRequest, seasons: list[Season]) -> tuple[Season, tuple[str, ...]]:
    lo, hi = HORIZON_RANGE
    if not lo <= req.horizon_hours <= hi:
        raise InvalidRequestError(f"horizon_hours must be in [{lo}, {hi}], got {req.horizon_hours}")
    clo, chi = COVERAGE_RANGE
    if not clo <= req.coverage <= chi:
        raise InvalidRequestError(f"coverage must be in [{clo}, {chi}], got {req.coverage}")
    if req.backtest_origin_step < 1:
        raise InvalidRequestError("backtest_origin_step must be >= 1")
    unknown = [m for m in req.models if m not in MODEL_FACTORIES]
    if unknown:
        raise InvalidRequestError(f"unknown models {unknown}; valid: {sorted(MODEL_FACTORIES)}")
    season = next((s for s in seasons if s.id == req.season_id), None)
    if season is None:
        raise InvalidRequestError(f"unknown season_id {req.season_id!r}")
    chosen = tuple(req.models) if req.models else tuple(MODEL_FACTORIES)
    names = tuple(dict.fromkeys(BASELINE_MODELS + chosen))  # baselines first, no duplicates
    return season, names


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _fit_selected(selection, series: pd.DataFrame, season_start: pd.Timestamp, horizon: int):
    """Refit members on the full series; return (point forecast, importance or None)."""
    preds, importance = [], None
    for name, weight in selection.members:
        model = make_model(name).fit(series, season_start)
        preds.append(weight * model.predict(horizon))
        if importance is None:
            importance = model.importance()
    return np.sum(preds, axis=0), importance


def _naive_selection(result, skill: float, reason: str):
    from .select import Selection
    return Selection("naive", (("naive", 1.0),), False, skill, result.predictions["naive"], reason)


def run_forecast(request: ForecastRequest, load_listings: ListingLoader, seasons: list[Season],
                 source: str = "unknown", log: Logger | None = None) -> dict:
    say = log or (lambda _s: None)
    season, model_names = _validate(request, seasons)
    say(f"loading listings for {request.item} in {season.name}")
    listings = load_listings(request.item, season.start, season.end)
    full_series = hourly_series(listings)
    series, warnings = prepare_target(full_series)
    filled = int((~series["observed"]).sum()) if "observed" in series.columns else 0
    if filled:
        warnings.append(f"{filled} gap hours forward-filled for model input (not scored)")
    if len(series) < MIN_HOURS:
        raise InsufficientDataError(
            f"need at least {MIN_HOURS} contiguous hours, have {len(series)} "
            f"(season {season.name}, {len(listings)} listings)")
    say(f"{len(series)} hourly points, backtesting {len(model_names)} models")

    result = run_backtest(series, season.start, model_names, request.horizon_hours,
                          origin_step=request.backtest_origin_step, log=say)
    warnings += list(result.warnings)
    warnings += [f"model {n} failed: {err}" for n, err in result.failures.items()]
    metrics = summarize(result)
    selection = select_model(result, metrics)
    say(f"selected {selection.name} (skill vs naive {selection.skill_vs_naive:+.1%})")

    if selection.reason:
        warnings.append(f"showing the naive forecast: {selection.reason}")
    try:
        point, importance = _fit_selected(selection, series, season.start, request.horizon_hours)
    except Exception as exc:
        warnings.append(f"{selection.name} failed on the full season ({type(exc).__name__}: {exc}); "
                        "falling back to the naive forecast")
        selection = _naive_selection(result, selection.skill_vs_naive, "selected model failed to refit")
        point, importance = _fit_selected(selection, series, season.start, request.horizon_hours)
    resid = result.actuals - selection.predictions
    lo, hi = quantile_bounds(resid, request.coverage)
    point = np.maximum(point, PRICE_FLOOR)
    lower, upper = apply_bounds(point, lo, hi, floor=PRICE_FLOOR)
    coverage = empirical_coverage(resid, request.coverage)
    if request.horizon_hours > result.horizon:
        warnings.append(f"intervals beyond h={result.horizon} reuse the last backtested width")

    last_ts = series.index[-1]
    current_offset = int((last_ts - season.start) / pd.Timedelta(hours=1))
    others = []
    for s in seasons:
        if s.id == season.id:
            continue
        other = hourly_series(load_listings(request.item, s.start, s.end))
        if not other.empty:
            others.append((s.id, s.name, other, s.start))
    references = reference_curves(current_offset, request.horizon_hours, others)

    tail = full_series.iloc[-TAIL_HOURS:]
    now = pd.Timestamp.now(tz="UTC")
    data_age_hours = float((now - last_ts) / pd.Timedelta(hours=1))
    if data_age_hours > STALE_HOURS:
        warnings.append(f"latest listing is {data_age_hours:.0f} hours old; the forecast starts from then, "
                        "not from now. Collect fresh data first.")
    run_id = f"{now:%Y%m%dT%H%M%S}-{request.item.lower().replace(' ', '_')}-{uuid.uuid4().hex[:6]}"
    return {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "request": {**asdict(request), "models": list(model_names)},
        "source": source,
        "season": {"id": season.id, "name": season.name, "start": season.start.isoformat(),
                   "end": season.end.isoformat() if season.end is not None else None},
        "series_tail": [
            {"ts": ts.isoformat(), "median_price": _num(r.median_price), "p25": _num(r.p25),
             "p75": _num(r.p75), "volume": int(r.volume), "listings": int(r.listings)}
            for ts, r in tail.iterrows()],
        "forecast": [
            {"ts": (last_ts + pd.Timedelta(hours=i + 1)).isoformat(), "h": i + 1,
             "pred": float(point[i]), "lower": float(lower[i]), "upper": float(upper[i])}
            for i in range(request.horizon_hours)],
        "selected": {"name": selection.name,
                     "members": [{"name": n, "weight": w} for n, w in selection.members]},
        "backtest": {
            "origins": len(result.origins), "horizon": result.horizon,
            "coverage": _num(coverage), "nominal_coverage": request.coverage,
            "models": [{"name": m.name, "mae_h1": _num(m.mae_h1), "mae_h24": _num(m.mae_h24),
                        "mae_mean": _num(m.mae_mean), "rmse_mean": _num(m.rmse_mean),
                        "mase_mean": _num(m.mase_mean), "direction_acc": _num(m.direction_acc)}
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
