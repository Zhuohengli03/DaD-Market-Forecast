"""Pick the model (or inverse-MAE ensemble) with the lowest backtest MAE; never worse than naive."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult, ModelMetrics

_EPS = 1e-9
MIN_SKILL = 0.02    # below this the win is within selection noise
MIN_ORIGINS = 5     # fewer backtest origins cannot support a claim


@dataclass(frozen=True)
class Selection:
    name: str
    members: tuple[tuple[str, float], ...]
    beats_baseline: bool
    skill_vs_naive: float
    predictions: np.ndarray  # backtest predictions of the selected model/ensemble
    reason: str = ""          # why naive was kept, when it was


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
    naive_pred = result.predictions["naive"]
    if len(result.origins) < MIN_ORIGINS:
        return Selection("naive", (("naive", 1.0),), False, float(skill), naive_pred,
                         f"只有 {len(result.origins)} 次回测，至少需要 {MIN_ORIGINS} 次才能判断哪个方法更准")
    if skill <= MIN_SKILL:
        why = "没有任何方法比\"沿用上一小时\"更准" if skill <= 0 else \
            f"最好的方法（{name}）只比基准准 {skill:.1%}，在随机波动范围内（需要超过 {MIN_SKILL:.0%}）"
        return Selection("naive", (("naive", 1.0),), False, float(skill), naive_pred, why)
    return Selection(name, members, True, float(skill), pred)
