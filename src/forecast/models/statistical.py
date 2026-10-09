from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import Forecaster, target


class ETSForecaster(Forecaster):
    name = "ets"
    min_seasonal_len = 72

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "ETSForecaster":
        y = target(series)
        seasonal = len(y) >= self.min_seasonal_len
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add" if seasonal else None,
                seasonal_periods=24 if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._fit.forecast(horizon), dtype=float)


class ARIMAForecaster(Forecaster):
    name = "arima"
    order = (1, 1, 1)

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "ARIMAForecaster":
        y = target(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = ARIMA(y, order=self.order).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._fit.forecast(horizon), dtype=float)
