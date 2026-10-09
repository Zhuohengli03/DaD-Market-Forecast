"""Pick the model (or inverse-MAE ensemble) with the lowest backtest MAE; never worse than naive."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult, ModelMetrics

_EPS = 1e-9


@dataclass(frozen=True)
class Selection:
    name: str
    members: tuple[tuple[str, float], ...]
    beats_baseline: bool
    skill_vs_naive: float
    predictions: np.ndarray  # backtest predictions of the selected model/ensemble


def _mae(result: BacktestResult, pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(result.actuals - pred)))


def select_model(result: BacktestResult, metrics: list[ModelMetrics], top_k: int = 3) -> Selection:
    by_name = {m.name: m for m in metrics}
    naive_mae = by_name["naive"].mae_mean
    ranked = sorted(metrics, key=lambda m: m.mae_mean)
    best = ranked[0]
    name, members, pred, mae = best.name, ((best.name, 1.0),), result.predictions[best.name], best.mae_mean

    candidates = ranked[:top_k]
    if len(candidates) >= 2:
        w = np.array([1.0 / max(m.mae_mean, _EPS) for m in candidates])
        w = w / w.sum()
        ens = sum(wi * result.predictions[m.name] for wi, m in zip(w, candidates))
        ens_mae = _mae(result, ens)
        if ens_mae < min(m.mae_mean for m in candidates):
            name, pred, mae = "ensemble", ens, ens_mae
            members = tuple((m.name, float(wi)) for m, wi in zip(candidates, w))

    skill = 1.0 - mae / naive_mae if naive_mae > 0 else 0.0
    if skill <= 0:
        return Selection("naive", (("naive", 1.0),), False, float(skill), result.predictions["naive"])
    return Selection(name, members, True, float(skill), pred)
