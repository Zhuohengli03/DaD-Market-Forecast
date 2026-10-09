import numpy as np
import pandas as pd
import pytest

from src.forecast.models import BASELINE_MODELS, MODEL_FACTORIES, make_model


def test_registry_contains_baselines():
    for n in BASELINE_MODELS:
        assert n in MODEL_FACTORIES
    with pytest.raises(KeyError):
        make_model("nope")


def test_naive_repeats_last(synthetic_series, season_start):
    m = make_model("naive").fit(synthetic_series, season_start)
    p = m.predict(5)
    assert p.shape == (5,) and np.allclose(p, synthetic_series["median_price"].iloc[-1])
    assert m.importance() is None


def test_seasonal_naive_tiles_last_day(synthetic_series, season_start):
    m = make_model("seasonal_naive_24").fit(synthetic_series, season_start)
    p = m.predict(30)
    last_day = synthetic_series["median_price"].iloc[-24:].to_numpy()
    assert np.allclose(p[:24], last_day) and np.allclose(p[24:30], last_day[:6])


def test_seasonal_naive_falls_back_when_short(season_start):
    idx = pd.date_range(season_start, periods=10, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": np.arange(10.0), "volume": 1, "listings": 1}, index=idx)
    p = make_model("seasonal_naive_24").fit(s, season_start).predict(3)
    assert np.allclose(p, 9.0)
