import numpy as np
import pytest

from src.forecast.intervals import apply_bounds, empirical_coverage, quantile_bounds


def test_quantile_bounds_widen_with_horizon():
    rng = np.random.default_rng(0)
    h = 10
    resid = np.stack([rng.normal(0, 1 + 0.3 * j, 200) for j in range(h)], axis=1)
    lo, hi = quantile_bounds(resid, 0.9)
    assert lo.shape == (h,) and hi.shape == (h,)
    assert (hi - lo)[-1] > (hi - lo)[0]
    assert lo[0] == pytest.approx(np.quantile(resid[:, 0], 0.05))


def test_quantile_bounds_borrows_neighbours_when_sparse():
    resid = np.zeros((3, 5))
    resid[:, 2] = [1.0, -1.0, 0.0]
    lo, hi = quantile_bounds(resid, 0.9, min_samples=10)
    assert lo[2] < 0 < hi[2]


def test_apply_bounds_extends_last_offset():
    preds = np.arange(5.0)
    lower, upper = apply_bounds(preds, np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
    assert np.allclose(lower, preds + np.array([-1, -2, -2, -2, -2]))
    assert np.allclose(upper, preds + np.array([1, 2, 2, 2, 2]))


def test_empirical_coverage_near_nominal():
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 1, (300, 4))
    cov = empirical_coverage(resid, 0.9)
    assert 0.85 < cov < 0.95
