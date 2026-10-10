import numpy as np

from src.forecast.backtest import BacktestResult, summarize
from src.forecast.select import select_model


def _result(preds: dict[str, np.ndarray], actuals: np.ndarray) -> BacktestResult:
    n, h = actuals.shape
    return BacktestResult(h, tuple(range(n)), actuals, actuals[:, 0] - 1, preds, {}, ())


def test_ensemble_chosen_when_it_beats_members():
    rng = np.random.default_rng(0)
    actual = np.zeros((50, 4))
    a = rng.normal(0, 1, (50, 4))
    b = -a + rng.normal(0, 0.1, (50, 4))          # strongly anti-correlated errors
    naive = rng.normal(0, 3, (50, 4))
    r = _result({"naive": naive, "a": a, "b": b}, actual)
    sel = select_model(r, summarize(r))
    assert sel.name == "ensemble"
    assert {m for m, _ in sel.members} >= {"a", "b"}
    assert sel.beats_baseline and sel.skill_vs_naive > 0
    assert sel.predictions.shape == (50, 4)


def test_single_model_chosen_when_ensemble_not_better():
    rng = np.random.default_rng(0)
    actual = np.zeros((50, 4))
    good = rng.normal(0, 0.1, (50, 4))
    bad = rng.normal(0, 5, (50, 4))
    naive = rng.normal(0, 3, (50, 4))
    r = _result({"naive": naive, "good": good, "bad": bad}, actual)
    sel = select_model(r, summarize(r))
    assert sel.name == "good" and sel.members == (("good", 1.0),)


def test_falls_back_to_naive_when_nothing_beats_it():
    rng = np.random.default_rng(0)
    actual = np.zeros((50, 4))
    naive = rng.normal(0, 0.1, (50, 4))
    worse = rng.normal(0, 2, (50, 4))
    r = _result({"naive": naive, "worse": worse}, actual)
    sel = select_model(r, summarize(r))
    assert sel.name == "naive" and not sel.beats_baseline and sel.skill_vs_naive <= 0


def test_small_margin_is_not_claimed_as_beating_baseline():
    rng = np.random.default_rng(0)
    actual = np.zeros((50, 4))
    naive = rng.normal(0, 1, (50, 4))
    almost = naive * 0.99                                   # 1% better: within noise
    r = _result({"naive": naive, "almost": almost}, actual)
    sel = select_model(r, summarize(r))
    assert sel.name == "naive" and not sel.beats_baseline and 0 < sel.skill_vs_naive < 0.02


def test_too_few_origins_is_not_claimed():
    rng = np.random.default_rng(0)
    actual = np.zeros((3, 4))
    naive = rng.normal(0, 3, (3, 4))
    good = rng.normal(0, 0.1, (3, 4))
    r = _result({"naive": naive, "good": good}, actual)
    sel = select_model(r, summarize(r))
    assert sel.name == "naive" and not sel.beats_baseline and "回测" in sel.reason


def test_reason_names_the_method_in_chinese():
    from src.forecast.select import method_name_zh
    assert method_name_zh("ridge") == "线性回归" and method_name_zh("ensemble") == "多模型加权"
    assert method_name_zh("mystery") == "mystery"
