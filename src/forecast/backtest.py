"""Rolling-origin backtest: for each origin, fit on rows [:o], forecast rows [o:o+h]."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .models import make_model

Logger = Callable[[str], None]


def default_origin_step(horizon: int) -> int:
    """Hours between backtest origins: windows overlapping by 3/4 at most, every 6 hours for short horizons."""
    return max(6, horizon // 4)


def independent_windows(origins, horizon: int) -> int:
    """How many backtest windows do not overlap each other: the real sample size behind the scores."""
    count, free_from = 0, None
    for o in origins:
        if free_from is None or o >= free_from:
            count, free_from = count + 1, o + horizon
    return count


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
        if shrunk < horizon and len(origins_for(shrunk)) > len(origins):
            warnings.append(
                f"数据只有 {n} 小时，回测评估长度从 {horizon} 小时缩短到 {shrunk} 小时，以获得足够的回测次数")
            h = shrunk
            origins = origins_for(h)
    if not origins:
        raise ValueError(f"数据只有 {n} 小时，太短，无法回测 {horizon} 小时的预测")
    if len(origins) < min_origins:
        warnings.append(f"只能做 {len(origins)} 次回测（建议至少 {min_origins} 次），结果参考价值有限")
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
    block_hours: int = 1           # hours per scored step (4 or 24 after aggregate_backtest)


def run_backtest(series: pd.DataFrame, season_start: pd.Timestamp, model_names: list[str] | tuple[str, ...],
                 horizon: int, origin_step: int = 6, min_origins: int = 5,
                 log: Logger | None = None, observed: np.ndarray | None = None,
                 on_model: Callable[[int, int, str], None] | None = None) -> BacktestResult:
    """`observed` marks hours with real listings; forward-filled hours are excluded from scoring."""
    y = series["median_price"].to_numpy(dtype=float)
    if observed is None and "observed" in series.columns:
        observed = series["observed"].to_numpy(dtype=bool)
    origins, h, warnings = plan_origins(len(series), horizon, origin_step, min_origins)
    actuals = np.stack([y[o:o + h] for o in origins])
    if observed is not None:
        mask = np.stack([~np.asarray(observed[o:o + h], dtype=bool) for o in origins])
        if mask.any():
            actuals = np.where(mask, np.nan, actuals)
            warnings.append(f"{int(mask.sum())} 个前向填充的时间点没有计入回测评分")
    last_values = np.array([y[o - 1] for o in origins])
    predictions: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for idx, name in enumerate(model_names, start=1):
        rows = []
        try:
            for o in origins:
                rows.append(make_model(name).fit(series.iloc[:o], season_start).predict(h))
            predictions[name] = np.stack(rows)
            if log:
                log(f"回测 {name}：完成 {len(origins)} 次")
            if on_model:
                on_model(idx, len(model_names), name)
        except Exception as exc:  # one bad model must not sink the run
            failures[name] = f"{type(exc).__name__}: {exc}"
            if log:
                log(f"回测 {name}：失败 {failures[name]}")
            if on_model:
                on_model(idx, len(model_names), name)
    return BacktestResult(h, tuple(origins), actuals, last_values, predictions, failures, tuple(warnings))


def _blocks(a: np.ndarray, k: int) -> np.ndarray:
    """(n, h) -> (n, h // k): the mean of each k-hour block, ignoring unscored hours."""
    n, h = a.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # an all-unscored block is NaN, as it should be
        return np.nanmean(a[:, : h // k * k].reshape(n, h // k, k), axis=2)


def aggregate_backtest(result: BacktestResult, y: np.ndarray, k: int) -> BacktestResult:
    """Score k-hour blocks instead of single hours (a steadier price).

    Forecasts and real prices become block averages, and the baseline becomes "the average of the last k hours
    before the origin", a fairer bar for a block than repeating one noisy hour.
    """
    if k == 1:
        return result
    last = np.array([np.nanmean(y[max(0, o - k):o]) for o in result.origins])
    preds = {name: _blocks(p, k) for name, p in result.predictions.items()}
    preds["naive"] = np.repeat(last[:, None], result.horizon // k, axis=1)
    return BacktestResult(result.horizon // k, result.origins, _blocks(result.actuals, k), last, preds,
                          result.failures, result.warnings, k)


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
    mdape_mean: float = float("nan")   # typical error: median of |error| / price, a rare spike barely moves it


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
        actual_dir, pred_dir = np.sign(result.actuals - last), np.sign(pred - last)
        # a step counts only when both the price and the forecast actually moved
        scored = ~np.isnan(result.actuals) & (actual_dir != 0) & (pred_dir != 0)
        direction = np.where(scored, actual_dir == pred_dir, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.abs(err) / np.where(result.actuals > 0, result.actuals, np.nan)
        out.append(ModelMetrics(
            name=name,
            mae_h1=float(per_h[0]),
            mae_h24=float(per_h[23]) if result.horizon >= 24 else float("nan"),
            mae_mean=mae,
            rmse_mean=rmse,
            mase_mean=mae / naive_mae if naive_mae and not np.isnan(naive_mae) else float("nan"),
            direction_acc=float(np.nanmean(direction)) if scored.any() else float("nan"),
            per_horizon_mae=tuple(float(v) for v in per_h),
            mdape_mean=float(np.nanmedian(ape)) if np.isfinite(ape).any() else float("nan"),
        ))
    return sorted(out, key=lambda m: m.mae_mean)
