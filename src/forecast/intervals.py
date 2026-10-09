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


def apply_bounds(preds: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(preds)
    if len(lo) < n:
        lo = np.concatenate([lo, np.full(n - len(lo), lo[-1])])
        hi = np.concatenate([hi, np.full(n - len(hi), hi[-1])])
    return preds + lo[:n], preds + hi[:n]


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
        hits.append(np.mean((row[ok] >= lo[ok]) & (row[ok] <= hi[ok])))
    return float(np.mean(hits))
