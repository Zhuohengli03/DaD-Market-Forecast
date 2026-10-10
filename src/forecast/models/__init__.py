from __future__ import annotations

from typing import Callable

from .base import Forecaster
from .baselines import NaiveForecaster, SeasonalNaiveForecaster
from .ml import make_hgb, make_rf, make_ridge
from .statistical import ARIMAForecaster, ETSForecaster

MODEL_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive_24": SeasonalNaiveForecaster,
    "ets": ETSForecaster,
    "arima": ARIMAForecaster,
    "ridge": make_ridge,
    "hgb": make_hgb,
    "rf": make_rf,
}
BASELINE_MODELS: tuple[str, ...] = ("naive", "seasonal_naive_24")


def register(name: str, factory: Callable[[], Forecaster]) -> None:
    MODEL_FACTORIES[name] = factory


def make_model(name: str) -> Forecaster:
    try:
        return MODEL_FACTORIES[name]()
    except KeyError:
        raise KeyError(f"unknown model {name!r}; valid: {sorted(MODEL_FACTORIES)}") from None


def default_models() -> tuple[str, ...]:
    return tuple(MODEL_FACTORIES)


__all__ = ["Forecaster", "MODEL_FACTORIES", "BASELINE_MODELS", "register", "make_model", "default_models"]
