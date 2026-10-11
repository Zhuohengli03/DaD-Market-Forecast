"""Pick the method on earlier backtests, judge it on later ones it was not picked on."""
import numpy as np
import pytest

from src.forecast.backtest import BacktestResult
from src.forecast.select import select_honestly


def _result(early_err: float, late_err: float, n: int = 20, h: int = 2, step: int = 2):
    actuals = np.full((n, h), 10.0)
    m = np.vstack([np.full((n // 2, h), 10.0 + early_err), np.full((n - n // 2, h), 10.0 + late_err)])
    preds = {"naive": np.full((n, h), 11.0), "seasonal_naive_24": np.full((n, h), 11.5), "m": m}
    return BacktestResult(h, tuple(range(100, 100 + n * step, step)), actuals, np.full(n, 10.0), preds, {}, ())


def test_a_method_that_only_won_early_is_not_used():
    # over all backtests "m" looks 45% better, but on the later half it is 20% worse than the baseline
    h = select_honestly(_result(early_err=0.1, late_err=1.2))
    assert h.selection.name == "naive" and h.selection.beats_baseline is False
    assert "后一半" in h.selection.reason
    assert h.in_sample is False and h.selection.skill_vs_naive == pytest.approx(-0.2)


def test_a_method_that_keeps_winning_is_used_and_its_skill_comes_from_the_later_half():
    h = select_honestly(_result(early_err=0.2, late_err=0.4))
    assert h.selection.name == "m" and h.selection.beats_baseline is True
    assert h.selection.skill_vs_naive == pytest.approx(0.6)          # 1 - 0.4 / 1.0, later half only
    assert h.evaluation.actuals.shape[0] == h.n_eval == 10           # intervals come from the later half
    assert h.selection.predictions.shape == h.evaluation.actuals.shape


def test_earlier_windows_that_reach_into_the_later_half_are_left_out():
    h = select_honestly(_result(0.2, 0.4, n=20, h=8, step=2))       # windows of 8h, 2h apart
    first_eval = h.evaluation.origins[0]
    assert h.n_select < 10 and all(o + 8 <= first_eval for o in h.selection_origins)


def test_too_few_backtests_to_split_fall_back_and_say_so():
    h = select_honestly(_result(0.2, 0.4, n=8))
    assert h.in_sample is True and h.selection.name == "m" and h.n_eval == 8


def test_block_windows_are_measured_in_hours_when_leaving_out_overlaps():
    r = _result(0.2, 0.4, n=20, h=2, step=4)
    blocks = BacktestResult(r.horizon, r.origins, r.actuals, r.last_values, r.predictions, {}, (), block_hours=4)
    h = select_honestly(blocks)                                     # 2 blocks of 4h = 8h windows, 4h apart
    assert all(o + 8 <= h.evaluation.origins[0] for o in h.selection_origins) and h.n_select == 9
