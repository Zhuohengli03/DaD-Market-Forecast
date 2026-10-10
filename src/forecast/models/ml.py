"""Recursive multi-step ML forecasters: predict t+1, append, recompute features, repeat."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import feature_columns, make_features
from .base import Forecaster

_HISTORY_COLS = ["median_price", "volume", "listings"]
_PROFILE_DAYS = 7


def _hourly_profile(series: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Mean volume/listings per hour-of-day over the last 7 days (future exogenous fill)."""
    tail = series.iloc[-24 * _PROFILE_DAYS:]
    grouped = tail.groupby(tail.index.hour)[["volume", "listings"]].mean()
    fallback = (float(tail["volume"].mean()), float(tail["listings"].mean()))
    return {h: (float(grouped.loc[h, "volume"]), float(grouped.loc[h, "listings"]))
            if h in grouped.index else fallback for h in range(24)}


class RecursiveMLForecaster(Forecaster):
    def __init__(self, name: str, make_estimator: Callable[[], object]):
        self.name = name
        self._make_estimator = make_estimator

    def fit(self, series: pd.DataFrame, season_start: pd.Timestamp) -> "RecursiveMLForecaster":
        self._history = series[_HISTORY_COLS].astype(float).copy()
        self._season_start = season_start
        X = make_features(series, season_start)
        y = series["median_price"].astype(float)
        mask = X.notna().all(axis=1) & y.notna()
        if mask.sum() < 10:
            raise ValueError(f"{self.name}: only {int(mask.sum())} usable training rows")
        self._X = X[mask].to_numpy()
        self._y = y[mask].to_numpy()
        with warnings.catch_warnings():
            # numpy 2.x on macOS Accelerate emits spurious "encountered in matmul" warnings
            # for transposed views; results are finite and identical to contiguous matmul.
            warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
            self._est = self._make_estimator().fit(self._X, self._y)
        self._profile = _hourly_profile(series)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        hist = self._history
        preds = np.empty(horizon)
        for i in range(horizon):
            next_ts = hist.index[-1] + pd.Timedelta(hours=1)
            vol, lst = self._profile[next_ts.hour]
            new_row = pd.DataFrame({"median_price": [np.nan], "volume": [vol], "listings": [lst]},
                                   index=pd.DatetimeIndex([next_ts], name=hist.index.name))
            extended = pd.concat([hist, new_row])
            x = make_features(extended, self._season_start).iloc[[-1]].fillna(0.0).to_numpy()
            preds[i] = float(self._est.predict(x)[0])
            extended.iloc[-1, extended.columns.get_loc("median_price")] = preds[i]
            hist = extended
        return preds

    def importance(self) -> list[tuple[str, float]] | None:
        names = feature_columns()
        est = self._est
        inner = est.steps[-1][1] if isinstance(est, Pipeline) else est
        if hasattr(inner, "feature_importances_"):
            values = np.asarray(inner.feature_importances_, dtype=float)
        elif hasattr(inner, "coef_"):
            values = np.abs(np.asarray(inner.coef_, dtype=float))
        else:
            values = permutation_importance(est, self._X, self._y, n_repeats=5,
                                            random_state=0).importances_mean
            values = np.clip(values, 0, None)
        pairs = list(zip(names, values.tolist()))
        return sorted(pairs, key=lambda p: p[1], reverse=True)


def make_ridge() -> RecursiveMLForecaster:
    return RecursiveMLForecaster("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))


def make_hgb() -> RecursiveMLForecaster:
    return RecursiveMLForecaster("hgb", lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0))


def make_rf() -> RecursiveMLForecaster:
    return RecursiveMLForecaster("rf", lambda: RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, random_state=0, n_jobs=-1))
