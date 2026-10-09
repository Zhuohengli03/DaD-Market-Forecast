from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    """Fit on an hourly series (columns median_price, volume, listings), predict h steps."""

    name: str = "base"

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> list[tuple[str, float]] | None:
        return None


def target(series: pd.DataFrame) -> np.ndarray:
    return series["median_price"].to_numpy(dtype=float)
