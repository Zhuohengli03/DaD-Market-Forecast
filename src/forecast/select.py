"""Pick the model (or inverse-MAE ensemble) with the lowest backtest MAE; never worse than naive."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult, ModelMetrics

_EPS = 1e-9
MIN_SKILL = 0.02    # below this the win is within selection noise
MIN_ORIGINS = 5     # fewer backtest origins cannot support a claim



METHOD_ZH = {"naive": "沿用上一小时", "seasonal_naive_24": "沿用昨天同一时刻", "ets": "指数平滑", "arima": "ARIMA 时间序列",
             "ridge": "线性回归", "hgb": "梯度提升树", "rf": "随机森林", "ensemble": "多模型加权"}


def method_name_zh(name: str) -> str:
    return METHOD_ZH.get(name, name)

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
            f"最好的方法（{method_name_zh(name)}）只比基准准 {skill:.1%}，在随机波动范围内（需要超过 {MIN_SKILL:.0%}）"
        return Selection("naive", (("naive", 1.0),), False, float(skill), naive_pred, why)
    return Selection(name, members, True, float(skill), pred)


@dataclass(frozen=True)
class HonestSelection:
    selection: Selection          # its predictions/skill cover `evaluation` only
    evaluation: BacktestResult    # the backtests the choice is judged on (and the band is built from)
    in_sample: bool               # True: too few backtests to split, judged on the same ones it was picked on
    n_select: int
    n_eval: int
    selection_origins: tuple[int, ...]


def _subset(result: BacktestResult, idx: np.ndarray) -> BacktestResult:
    return BacktestResult(result.horizon, tuple(np.asarray(result.origins)[idx].tolist()), result.actuals[idx],
                          result.last_values[idx], {n: p[idx] for n, p in result.predictions.items()},
                          result.failures, result.warnings, result.block_hours)


def select_honestly(result: BacktestResult, top_k: int = 3) -> HonestSelection:
    """Pick the method (and ensemble weights) on the earlier half of the backtests, judge it on the later half.

    Earlier windows that reach into the later half are left out, so no later price helps the pick. When there are
    too few backtests to split, it falls back to picking and judging on the same ones (in_sample=True).
    """
    from .backtest import summarize
    origins = np.asarray(result.origins)
    n = len(origins)
    half = n // 2
    first_eval = origins[half] if n else 0
    pick_idx = np.flatnonzero((np.arange(n) < half) & (origins + result.horizon * result.block_hours <= first_eval))
    eval_idx = np.arange(half, n)
    if len(pick_idx) < MIN_ORIGINS or len(eval_idx) < MIN_ORIGINS:
        sel = select_model(result, summarize(result), top_k)
        return HonestSelection(sel, result, True, n, n, tuple(origins.tolist()))
    early, late = _subset(result, pick_idx), _subset(result, eval_idx)
    picked = select_model(early, summarize(early), top_k)
    pred = sum(w * late.predictions[m] for m, w in picked.members)
    naive_mae = _mae(late, late.predictions["naive"])
    skill = 1.0 - _mae(late, pred) / naive_mae if naive_mae > 0 else 0.0
    if picked.name == "naive":
        sel = Selection("naive", (("naive", 1.0),), False, float(skill), late.predictions["naive"], picked.reason)
    elif skill <= MIN_SKILL:
        why = (f"前一半回测挑出的{method_name_zh(picked.name)}，在后一半回测里没有比基准明显更准"
               f"（{skill:+.1%}，需要超过 {MIN_SKILL:.0%}）")
        sel = Selection("naive", (("naive", 1.0),), False, float(skill), late.predictions["naive"], why)
    else:
        sel = Selection(picked.name, picked.members, True, float(skill), pred)
    return HonestSelection(sel, late, False, len(pick_idx), len(eval_idx), tuple(origins[pick_idx].tolist()))
