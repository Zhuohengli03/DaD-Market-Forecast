import numpy as np
import pandas as pd
import pytest

from src.forecast.features import feature_columns
from src.forecast.models import MODEL_FACTORIES, make_model

ALL = ["ets", "arima", "ridge", "hgb", "rf"]


def test_models_registered():
    for n in ALL:
        assert n in MODEL_FACTORIES


@pytest.mark.parametrize("name", ALL)
def test_fit_predict_shape_and_finite(name, synthetic_series, season_start):
    m = make_model(name).fit(synthetic_series, season_start)
    p = m.predict(48)
    assert p.shape == (48,) and np.isfinite(p).all()


@pytest.mark.parametrize("name", ["ridge", "hgb", "rf"])
def test_ml_recursion_is_deterministic_and_tracks_level(name, synthetic_series, season_start):
    m = make_model(name).fit(synthetic_series, season_start)
    p1, p2 = m.predict(24), m.predict(24)
    assert np.allclose(p1, p2)
    level = synthetic_series["median_price"].iloc[-24:].mean()
    assert abs(p1.mean() - level) < 5


@pytest.mark.parametrize("name", ["ridge", "hgb", "rf"])
def test_ml_importance_covers_all_features(name, synthetic_series, season_start):
    imp = make_model(name).fit(synthetic_series, season_start).importance()
    assert imp is not None
    assert sorted(n for n, _ in imp) == sorted(feature_columns())
    assert all(v >= 0 for _, v in imp)
    assert [v for _, v in imp] == sorted((v for _, v in imp), reverse=True)


def test_ets_short_series_without_seasonality(season_start):
    idx = pd.date_range(season_start, periods=40, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": np.linspace(10, 12, 40), "volume": 1, "listings": 1}, index=idx)
    p = make_model("ets").fit(s, season_start).predict(5)
    assert np.isfinite(p).all()
