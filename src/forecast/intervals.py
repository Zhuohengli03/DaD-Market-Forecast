"""Prediction intervals from per-horizon empirical residual quantiles."""
from __future__ import annotations

import numpy as np

from .backtest import BacktestResult


def residuals(result: BacktestResult, name: str) -> np.ndarray:
    return result.actuals - result.predictions[name]


def _pool(resid: np.ndarray, j: int, min_samples: int) -> np.ndarray:
    h = resid.shape[1]
    pool = resid[:, j]
    k = 1
    while pool.size < min_samples and (j - k >= 0 or j + k < h):
        parts = [pool]
        if j - k >= 0:
            parts.append(resid[:, j - k])
        if j + k < h:
            parts.append(resid[:, j + k])
        pool = np.concatenate(parts)
        k += 1
    return pool[~np.isnan(pool)]


def quantile_bounds(resid: np.ndarray, coverage: float, min_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - coverage) / 2.0
    h = resid.shape[1]
    lo, hi = np.empty(h), np.empty(h)
    for j in range(h):
        pool = _pool(resid, j, min_samples)
        lo[j] = np.quantile(pool, alpha)
        hi[j] = np.quantile(pool, 1.0 - alpha)
    return lo, hi


def apply_bounds(preds: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add per-horizon offsets to point forecasts; `floor` clamps both bounds (prices are >= 0)."""
    n = len(preds)
    if len(lo) < n:
        lo = np.concatenate([lo, np.full(n - len(lo), lo[-1])])
        hi = np.concatenate([hi, np.full(n - len(hi), hi[-1])])
    lower, upper = preds + lo[:n], preds + hi[:n]
    if floor is not None:
        lower, upper = np.maximum(lower, floor), np.maximum(upper, floor)
    return lower, upper


def empirical_coverage(resid: np.ndarray, coverage: float, min_samples: int = 10) -> float:
    """Leave-one-origin-out coverage of the quantile intervals within the backtest."""
    n = resid.shape[0]
    if n < 3:
        return float("nan")
    hits = []
    for i in range(n):
        lo, hi = quantile_bounds(np.delete(resid, i, axis=0), coverage, min_samples)
        row = resid[i]
        ok = ~np.isnan(row)
        if ok.any():
            hits.append(np.mean((row[ok] >= lo[ok]) & (row[ok] <= hi[ok])))
    return float(np.mean(hits)) if hits else float("nan")


def relative_residuals(actuals: np.ndarray, preds: np.ndarray, last_values: np.ndarray) -> np.ndarray:
    """Errors as a share of the price at each origin, so a band learned at 95 still fits at 20."""
    last = np.asarray(last_values, dtype=float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(last > 0, (actuals - preds) / last, np.nan)


def apply_relative_bounds(preds: np.ndarray, lo: np.ndarray, hi: np.ndarray, last_value: float,
                          floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Relative quantile offsets scaled by the latest price, then added to the point forecast."""
    return apply_bounds(preds, lo * last_value, hi * last_value, floor=floor)
