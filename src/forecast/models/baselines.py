from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, target


class NaiveForecaster(Forecaster):
    name = "naive"

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "NaiveForecaster":
        self._last = float(target(series)[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last)


class SeasonalNaiveForecaster(Forecaster):
    name = "seasonal_naive_24"
    period = 24

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "SeasonalNaiveForecaster":
        y = target(series)
        self._pattern = y[-self.period:] if len(y) >= self.period else np.array([y[-1]])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / len(self._pattern)))
        return np.tile(self._pattern, reps)[:horizon]
