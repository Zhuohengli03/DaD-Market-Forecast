"""Rolling-origin backtest: for each origin, fit on rows [:o], forecast rows [o:o+h]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .models import make_model

Logger = Callable[[str], None]


def plan_origins(n: int, horizon: int, step: int, min_origins: int = 5,
                 min_horizon: int = 24) -> tuple[list[int], int, list[str]]:
    start = max(48, int(0.6 * n))

    def origins_for(h: int) -> list[int]:
        return list(range(start, n - h + 1, step))

    warnings: list[str] = []
    h = horizon
    origins = origins_for(h)
    if len(origins) < min_origins:
        shrunk = min(horizon, max(min_horizon, n - start - (min_origins - 1) * step))
        if shrunk < horizon and len(origins_for(shrunk)) >= min_origins:
            warnings.append(
                f"evaluation horizon shrunk from {horizon}h to {shrunk}h to obtain "
                f"{min_origins} backtest origins (series has {n} hours)")
            h = shrunk
            origins = origins_for(h)
    if not origins:
        raise ValueError(f"series of {n} hours is too short to backtest a {horizon}h horizon")
    if len(origins) < min_origins:
        warnings.append(f"only {len(origins)} backtest origins available (wanted {min_origins})")
    return origins, h, warnings


@dataclass(frozen=True)
class BacktestResult:
    horizon: int
    origins: tuple[int, ...]
    actuals: np.ndarray            # (n_origins, horizon)
    last_values: np.ndarray        # (n_origins,) value at origin-1
    predictions: dict[str, np.ndarray]
    failures: dict[str, str]
    warnings: tuple[str, ...]


def run_backtest(series: pd.DataFrame, season_start: pd.Timestamp, model_names: list[str] | tuple[str, ...],
                 horizon: int, origin_step: int = 6, min_origins: int = 5,
                 log: Logger | None = None) -> BacktestResult:
    y = series["median_price"].to_numpy(dtype=float)
    origins, h, warnings = plan_origins(len(series), horizon, origin_step, min_origins)
    actuals = np.stack([y[o:o + h] for o in origins])
    last_values = np.array([y[o - 1] for o in origins])
    predictions: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for name in model_names:
        rows = []
        try:
            for o in origins:
                rows.append(make_model(name).fit(series.iloc[:o], season_start).predict(h))
            predictions[name] = np.stack(rows)
            if log:
                log(f"backtest {name}: {len(origins)} origins done")
        except Exception as exc:  # one bad model must not sink the run
            failures[name] = f"{type(exc).__name__}: {exc}"
            if log:
                log(f"backtest {name}: FAILED {failures[name]}")
    return BacktestResult(h, tuple(origins), actuals, last_values, predictions, failures, tuple(warnings))


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    mae_h1: float
    mae_h24: float
    mae_mean: float
    rmse_mean: float
    mase_mean: float
    direction_acc: float
    per_horizon_mae: tuple[float, ...]


def summarize(result: BacktestResult) -> list[ModelMetrics]:
    naive_mae = float(np.nanmean(np.abs(result.actuals - result.predictions["naive"]))) \
        if "naive" in result.predictions else float("nan")
    out: list[ModelMetrics] = []
    for name, pred in result.predictions.items():
        err = result.actuals - pred
        per_h = np.nanmean(np.abs(err), axis=0)
        mae = float(np.nanmean(np.abs(err)))
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        last = result.last_values[:, None]
        direction = np.sign(pred - last) == np.sign(result.actuals - last)
        out.append(ModelMetrics(
            name=name,
            mae_h1=float(per_h[0]),
            mae_h24=float(per_h[23]) if result.horizon >= 24 else float("nan"),
            mae_mean=mae,
            rmse_mean=rmse,
            mase_mean=mae / naive_mae if naive_mae and not np.isnan(naive_mae) else float("nan"),
            direction_acc=float(np.nanmean(direction)),
            per_horizon_mae=tuple(float(v) for v in per_h),
        ))
    return sorted(out, key=lambda m: m.mae_mean)
