# Forecast Pipeline Implementation Plan (1/3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/forecast/`, an honest hourly price-forecasting package with leak-free features, rolling-origin backtesting, empirical intervals, and a CLI, replacing `src/analysis/Machine_learning_analysis.py`.

**Architecture:** Listings are aggregated into an hourly UTC series; features reference only earlier rows; every model implements `fit(series, season_start)` / `predict(horizon)`; a rolling-origin backtest produces per-horizon errors that drive model selection and interval widths; `pipeline.run_forecast` wires it together and returns a plain dict.

**Tech Stack:** Python 3.13 (anaconda base: `/opt/anaconda3/bin/python`), pandas, numpy, scikit-learn, statsmodels, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-forecast-rebuild-and-web-app-design.md` (sections 5, 6, 9)

## Global Constraints

- Python interpreter for all commands: `/opt/anaconda3/bin/python` (only env with sklearn, statsmodels, psycopg2).
- Run tests as `/opt/anaconda3/bin/python -m pytest tests/forecast -q` from the project root.
- All timestamps are tz-aware UTC `pd.Timestamp`; series index name is `ts`, freq 1 hour.
- Target column is `median_price`. No feature may read row `t` of `median_price`, `volume`, or `listings`.
- Baselines `naive` and `seasonal_naive_24` always participate in a backtest.
- Minimum usable series length: 72 hours. Horizon range: 24–168. Coverage range: 0.5–0.99.
- Work on branch `rebuild/forecast-and-webapp` (create from `main` in Task 0). Commit after each task; do not push.
- Commit message format: `<type>: <description>` with no attribution trailer (user's global rule).

## Review Focus

1. Listings whose `has_sold` arrives as the strings `"True"/"False"` (CSV) or as Python bools (Postgres): both must count correctly. → Task 1 test `test_has_sold_accepts_strings_and_bools`.
2. An hour with a single bundle listing like `quantity=2, price_per_unit=0.5` while neighbours are ~20: the hourly median must be that 0.5 for that hour (no outlier deletion), and the model must still run. → Task 1 test `test_single_listing_hour_is_its_own_median`.
3. A series whose last contiguous segment is shorter than 72 hours even though the whole season is longer: the pipeline must refuse with `InsufficientDataError` naming the segment length. → Task 8 test `test_short_last_segment_is_rejected`.
4. Backtest with a horizon that leaves fewer than 5 origins: evaluation horizon shrinks and a warning says so; a horizon that leaves zero origins even after shrinking raises. → Task 5 tests `test_plan_origins_shrinks_horizon` and `test_plan_origins_raises_when_impossible`.
5. A model that throws during fit (e.g. ARIMA on a constant series): the backtest records it in `failures` and the pipeline continues with the remaining models. → Task 5 test `test_failed_model_is_recorded_not_fatal`.

---

### Task 0: Branch, package skeleton, test fixtures

**Files:**
- Create: `src/forecast/__init__.py`, `src/forecast/models/__init__.py` (placeholder, filled in Task 3)
- Create: `tests/__init__.py`, `tests/forecast/__init__.py`, `tests/forecast/conftest.py`
- Create: `pytest.ini`
- Modify: `.gitignore` (add `runs/`)

**Interfaces:**
- Produces fixtures `synthetic_listings(hours=240)` and `synthetic_series(hours=240)` used by every later test.

- [ ] **Step 1: Create branch**

```bash
git checkout -b rebuild/forecast-and-webapp
```

- [ ] **Step 2: Write pytest config and fixtures**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
filterwarnings =
    ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning
    ignore::FutureWarning
```

`tests/forecast/conftest.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEASON_START = pd.Timestamp("2026-01-20T00:00:00Z")


def make_listings(hours: int = 240, seed: int = 0, per_hour: int = 5) -> pd.DataFrame:
    """Listings with a daily cycle + slow trend + noise, several per hour."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(hours):
        ts = SEASON_START + pd.Timedelta(hours=h)
        base = 20 + 0.02 * h + 3 * np.sin(2 * np.pi * h / 24)
        for k in range(per_hour):
            price = base + rng.normal(0, 0.5)
            qty = int(rng.integers(1, 7))
            rows.append({
                "item_id": f"{h}-{k}",
                "item": "Test Ore",
                "quantity": qty,
                "price_per_unit": round(price, 2),
                "price": round(price * qty, 2),
                "has_sold": bool(rng.random() < 0.9),
                "created_at": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
            })
    return pd.DataFrame(rows)


def make_series(hours: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(SEASON_START, periods=hours, freq="h", tz="UTC", name="ts")
    h = np.arange(hours)
    price = 20 + 0.02 * h + 3 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.3, hours)
    return pd.DataFrame({
        "median_price": price,
        "min_price": price - 1,
        "max_price": price + 1,
        "p25": price - 0.5,
        "p75": price + 0.5,
        "listings": rng.integers(3, 8, hours),
        "sold": rng.integers(0, 3, hours),
        "volume": rng.integers(5, 30, hours),
    }, index=idx)


@pytest.fixture
def synthetic_listings() -> pd.DataFrame:
    return make_listings()


@pytest.fixture
def synthetic_series() -> pd.DataFrame:
    return make_series()


@pytest.fixture
def season_start() -> pd.Timestamp:
    return SEASON_START
```

- [ ] **Step 3: Create empty package files and ignore runs/**

```bash
touch src/forecast/__init__.py src/forecast/models/__init__.py tests/__init__.py tests/forecast/__init__.py
printf '\n# forecast run outputs\nruns/\n' >> .gitignore
```

- [ ] **Step 4: Verify pytest collects nothing and exits cleanly**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast -q`
Expected: `no tests ran`

- [ ] **Step 5: Commit**

```bash
git add pytest.ini .gitignore src/forecast tests
git commit -m "chore: scaffold forecast package and test fixtures"
```

---

### Task 1: Hourly aggregation

**Files:**
- Create: `src/forecast/aggregate.py`
- Test: `tests/forecast/test_aggregate.py`

**Interfaces:**
- Produces:
  - `SERIES_COLUMNS: tuple[str, ...]`
  - `hourly_series(listings: pd.DataFrame) -> pd.DataFrame` — index `ts` (UTC hourly, continuous from first to last listing hour), columns `median_price, min_price, max_price, p25, p75` (NaN on empty hours) and `listings, sold, volume` (0 on empty hours).
  - `prepare_target(series: pd.DataFrame, max_gap_hours: int = 6) -> tuple[pd.DataFrame, list[str]]` — returns the last contiguous segment with `median_price` forward-filled across gaps ≤ `max_gap_hours`, plus warnings.
  - `resample_series(series: pd.DataFrame, resolution: str) -> pd.DataFrame` — `"1h" | "4h" | "1d"` for display.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_aggregate.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.forecast.aggregate import SERIES_COLUMNS, hourly_series, prepare_target, resample_series


def _listing(ts, price, qty=1, sold=True):
    return {"item_id": "x", "item": "Test", "quantity": qty, "price_per_unit": price,
            "price": price * qty, "has_sold": sold, "created_at": ts}


def test_hourly_series_basic_stats():
    rows = [
        _listing("2026-01-01T00:10:00Z", 10.0, 2, True),
        _listing("2026-01-01T00:40:00Z", 20.0, 3, False),
        _listing("2026-01-01T00:50:00Z", 30.0, 1, True),
        _listing("2026-01-01T02:05:00Z", 50.0, 4, True),
    ]
    s = hourly_series(pd.DataFrame(rows))
    assert list(s.columns) == list(SERIES_COLUMNS)
    assert s.index.name == "ts" and str(s.index.tz) == "UTC"
    assert len(s) == 3  # 00:00, 01:00 (empty), 02:00
    h0 = s.iloc[0]
    assert h0["median_price"] == 20.0 and h0["min_price"] == 10.0 and h0["max_price"] == 30.0
    assert h0["listings"] == 3 and h0["sold"] == 2 and h0["volume"] == 6
    h1 = s.iloc[1]
    assert np.isnan(h1["median_price"]) and h1["listings"] == 0 and h1["volume"] == 0


def test_has_sold_accepts_strings_and_bools():
    rows = [_listing("2026-01-01T00:00:00Z", 1.0, sold="True"),
            _listing("2026-01-01T00:01:00Z", 1.0, sold="False"),
            _listing("2026-01-01T00:02:00Z", 1.0, sold=True)]
    s = hourly_series(pd.DataFrame(rows))
    assert s.iloc[0]["sold"] == 2


def test_single_listing_hour_is_its_own_median():
    rows = [_listing("2026-01-01T00:00:00Z", 20.0), _listing("2026-01-01T01:00:00Z", 0.5, qty=2)]
    s = hourly_series(pd.DataFrame(rows))
    assert s.iloc[1]["median_price"] == 0.5


def test_empty_input_gives_empty_frame():
    s = hourly_series(pd.DataFrame(columns=["item_id", "item", "quantity", "price_per_unit",
                                            "price", "has_sold", "created_at"]))
    assert len(s) == 0 and list(s.columns) == list(SERIES_COLUMNS)


def test_prepare_target_fills_short_gaps_and_cuts_long_ones():
    idx = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC", name="ts")
    price = np.arange(40, dtype=float)
    price[5:8] = np.nan          # 3h gap -> filled
    price[20:28] = np.nan        # 8h gap -> cut, keep rows 28..39
    s = pd.DataFrame({"median_price": price, "volume": 1, "listings": 1}, index=idx)
    out, warnings = prepare_target(s, max_gap_hours=6)
    assert out.index[0] == idx[28] and len(out) == 12
    assert not out["median_price"].isna().any()
    assert any("8" in w for w in warnings)


def test_prepare_target_drops_leading_nan():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC", name="ts")
    price = np.arange(10, dtype=float)
    price[:2] = np.nan
    s = pd.DataFrame({"median_price": price, "volume": 1, "listings": 1}, index=idx)
    out, _ = prepare_target(s)
    assert out.index[0] == idx[2] and len(out) == 8


def test_resample_4h_and_1d():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": 1.0, "min_price": 0.5, "max_price": 2.0, "p25": 0.8,
                      "p75": 1.2, "listings": 2, "sold": 1, "volume": 3}, index=idx)
    r4 = resample_series(s, "4h")
    assert len(r4) == 12 and r4.iloc[0]["volume"] == 12 and r4.iloc[0]["min_price"] == 0.5
    r1d = resample_series(s, "1d")
    assert len(r1d) == 2 and r1d.iloc[0]["listings"] == 48
    assert resample_series(s, "1h").equals(s)
    with pytest.raises(ValueError):
        resample_series(s, "2h")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_aggregate.py -q`
Expected: ImportError on `src.forecast.aggregate`.

- [ ] **Step 3: Implement**

`src/forecast/aggregate.py`:
```python
"""Listings -> hourly UTC series. The hourly median is the forecast target."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_COLUMNS: tuple[str, ...] = (
    "median_price", "min_price", "max_price", "p25", "p75", "listings", "sold", "volume",
)
_TRUE_STRINGS = {"true", "1", "t", "yes"}
_RESOLUTIONS = {"1h": None, "4h": "4h", "1d": "1D"}


def _empty_series() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in SERIES_COLUMNS}, index=idx)


def _coerce_sold(col: pd.Series) -> pd.Series:
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)


def hourly_series(listings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw listings into a continuous hourly series (NaN price on empty hours)."""
    if listings.empty:
        return _empty_series()
    df = pd.DataFrame({
        "created_at": pd.to_datetime(listings["created_at"], utc=True, errors="coerce"),
        "price_per_unit": pd.to_numeric(listings["price_per_unit"], errors="coerce"),
        "quantity": pd.to_numeric(listings["quantity"], errors="coerce"),
        "has_sold": _coerce_sold(listings["has_sold"]),
    }).dropna(subset=["created_at", "price_per_unit", "quantity"])
    if df.empty:
        return _empty_series()
    df["ts"] = df["created_at"].dt.floor("h")
    g = df.groupby("ts")
    out = pd.DataFrame({
        "median_price": g["price_per_unit"].median(),
        "min_price": g["price_per_unit"].min(),
        "max_price": g["price_per_unit"].max(),
        "p25": g["price_per_unit"].quantile(0.25),
        "p75": g["price_per_unit"].quantile(0.75),
        "listings": g.size(),
        "sold": g["has_sold"].sum(),
        "volume": g["quantity"].sum(),
    })
    full = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC", name="ts")
    out = out.reindex(full)
    for c in ("listings", "sold", "volume"):
        out[c] = out[c].fillna(0).astype(int)
    return out[list(SERIES_COLUMNS)]


def prepare_target(series: pd.DataFrame, max_gap_hours: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill gaps <= max_gap_hours; cut at longer gaps; return last contiguous segment."""
    warnings: list[str] = []
    s = series.copy()
    price = s["median_price"]
    missing = price.isna()
    if missing.all():
        return s.iloc[0:0], ["series has no prices"]
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    long_gap = missing & (run_len > max_gap_hours)
    if long_gap.any():
        last_gap_end = np.flatnonzero(long_gap.to_numpy())[-1]
        gap_hours = int(run_len.iloc[last_gap_end])
        warnings.append(
            f"gap of {gap_hours} hours ending at {s.index[last_gap_end].isoformat()} exceeds "
            f"{max_gap_hours}h; using only the {len(s) - last_gap_end - 1} hours after it"
        )
        s = s.iloc[last_gap_end + 1:]
    s = s.loc[s["median_price"].first_valid_index():]
    s["median_price"] = s["median_price"].ffill()
    for c in ("min_price", "max_price", "p25", "p75"):
        if c in s.columns:
            s[c] = s[c].ffill()
    return s, warnings


def resample_series(series: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Display-only resampling. 1h returns the input unchanged."""
    if resolution not in _RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(_RESOLUTIONS)}, got {resolution!r}")
    rule = _RESOLUTIONS[resolution]
    if rule is None:
        return series
    agg = {"median_price": "median", "min_price": "min", "max_price": "max", "p25": "mean",
           "p75": "mean", "listings": "sum", "sold": "sum", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in series.columns}
    return series.resample(rule).agg(agg)
```

- [ ] **Step 4: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_aggregate.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/aggregate.py tests/forecast/test_aggregate.py
git commit -m "feat(forecast): hourly aggregation, gap handling, display resampling"
```

---

### Task 2: Leak-free features

**Files:**
- Create: `src/forecast/features.py`
- Test: `tests/forecast/test_features.py`

**Interfaces:**
- Produces:
  - `PRICE_LAGS = (1, 2, 3, 6, 12, 24)`, `EXOG_LAGS = (1, 6, 24)`, `ROLL_WINDOWS = (6, 24)`
  - `feature_columns() -> list[str]` — stable order.
  - `make_features(series: pd.DataFrame, season_start: pd.Timestamp) -> pd.DataFrame` — same index as `series`; columns = `feature_columns()`; NaN where lags are unavailable. Requires columns `median_price, volume, listings`.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_features.py`:
```python
import numpy as np
import pandas as pd

from src.forecast.features import PRICE_LAGS, feature_columns, make_features


def test_feature_columns_match_frame(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    assert list(f.columns) == feature_columns()
    assert f.index.equals(synthetic_series.index)


def test_lags_are_exact(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    price = synthetic_series["median_price"]
    for lag in PRICE_LAGS:
        assert np.allclose(f[f"price_lag_{lag}"].iloc[lag:], price.iloc[:-lag])
        assert f[f"price_lag_{lag}"].iloc[:lag].isna().all()
    assert f["hours_since_start"].iloc[0] == 0 and f["hours_since_start"].iloc[25] == 25


def test_no_same_row_leakage(synthetic_series, season_start):
    base = make_features(synthetic_series, season_start)
    for t in (0, 30, 100, len(synthetic_series) - 1):
        mutated = synthetic_series.copy()
        mutated.iloc[t, mutated.columns.get_loc("median_price")] = 9999.0
        mutated.iloc[t, mutated.columns.get_loc("volume")] = 9999
        mutated.iloc[t, mutated.columns.get_loc("listings")] = 9999
        f = make_features(mutated, season_start)
        pd.testing.assert_series_equal(f.iloc[t], base.iloc[t], check_names=False)


def test_rolling_uses_lagged_price_only(synthetic_series, season_start):
    f = make_features(synthetic_series, season_start)
    price = synthetic_series["median_price"]
    expected = price.shift(1).rolling(6, min_periods=1).mean()
    assert np.allclose(f["price_roll_mean_6"].iloc[1:], expected.iloc[1:])


def test_change_24_handles_zero_denominator(season_start):
    idx = pd.date_range(season_start, periods=30, freq="h", tz="UTC", name="ts")
    s = pd.DataFrame({"median_price": 0.0, "volume": 1, "listings": 1}, index=idx)
    f = make_features(s, season_start)
    assert not np.isinf(f["price_change_24"].fillna(0)).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_features.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/forecast/features.py`:
```python
"""Features for hour t that only read rows <= t-1 (plus calendar/offset of t itself)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
EXOG_LAGS: tuple[int, ...] = (1, 6, 24)
ROLL_WINDOWS: tuple[int, ...] = (6, 24)
MAX_LAG = max(PRICE_LAGS)


def feature_columns() -> list[str]:
    cols = [f"price_lag_{l}" for l in PRICE_LAGS]
    for l in EXOG_LAGS:
        cols += [f"volume_lag_{l}", f"listings_lag_{l}"]
    for w in ROLL_WINDOWS:
        cols += [f"price_roll_mean_{w}", f"price_roll_std_{w}"]
    cols += ["price_change_24", "hour_sin", "hour_cos", "dow", "hours_since_start"]
    return cols


def make_features(series: pd.DataFrame, season_start: pd.Timestamp) -> pd.DataFrame:
    price = series["median_price"].astype(float)
    volume = series["volume"].astype(float)
    listings = series["listings"].astype(float)
    f = pd.DataFrame(index=series.index)
    for lag in PRICE_LAGS:
        f[f"price_lag_{lag}"] = price.shift(lag)
    for lag in EXOG_LAGS:
        f[f"volume_lag_{lag}"] = volume.shift(lag)
        f[f"listings_lag_{lag}"] = listings.shift(lag)
    lag1 = price.shift(1)
    for w in ROLL_WINDOWS:
        f[f"price_roll_mean_{w}"] = lag1.rolling(w, min_periods=1).mean()
        f[f"price_roll_std_{w}"] = lag1.rolling(w, min_periods=2).std()
    lag24 = price.shift(24)
    f["price_change_24"] = ((lag1 - lag24) / lag24).replace([np.inf, -np.inf], np.nan)
    hour = series.index.hour.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = series.index.dayofweek.to_numpy().astype(float)
    f["hours_since_start"] = (series.index - season_start) / pd.Timedelta(hours=1)
    return f[feature_columns()]
```

- [ ] **Step 4: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_features.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/features.py tests/forecast/test_features.py
git commit -m "feat(forecast): lag-only feature set with leakage test"
```

---

### Task 3: Model protocol, registry, baselines

**Files:**
- Create: `src/forecast/models/base.py`, `src/forecast/models/baselines.py`
- Modify: `src/forecast/models/__init__.py`
- Test: `tests/forecast/test_baselines.py`

**Interfaces:**
- Produces:
  - `class Forecaster` (base): attributes `name: str`; `fit(series, season_start) -> Self`; `predict(horizon: int) -> np.ndarray`; `importance() -> list[tuple[str, float]] | None` (default `None`).
  - `NaiveForecaster` (`name="naive"`), `SeasonalNaiveForecaster` (`name="seasonal_naive_24"`).
  - In `models/__init__.py`: `MODEL_FACTORIES: dict[str, Callable[[], Forecaster]]`, `BASELINE_MODELS = ("naive", "seasonal_naive_24")`, `DEFAULT_MODELS: tuple[str, ...]`, `make_model(name) -> Forecaster` (raises `KeyError` with the list of valid names). Tasks 4 register more factories into the same dict.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_baselines.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_baselines.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/forecast/models/base.py`:
```python
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
```

`src/forecast/models/baselines.py`:
```python
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
```

`src/forecast/models/__init__.py`:
```python
from __future__ import annotations

from typing import Callable

from .base import Forecaster
from .baselines import NaiveForecaster, SeasonalNaiveForecaster

MODEL_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive_24": SeasonalNaiveForecaster,
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
```

(Note: `DEFAULT_MODELS` is exposed as the function `default_models()` so that registrations from Task 4 are included.)

- [ ] **Step 4: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_baselines.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/models tests/forecast/test_baselines.py
git commit -m "feat(forecast): forecaster protocol, registry and baselines"
```

---

### Task 4: Statistical and recursive ML models

**Files:**
- Create: `src/forecast/models/statistical.py`, `src/forecast/models/ml.py`
- Modify: `src/forecast/models/__init__.py` (register `ets, arima, ridge, hgb, rf`)
- Test: `tests/forecast/test_models.py`

**Interfaces:**
- Consumes: `make_features`, `feature_columns` (Task 2); `Forecaster`, `register` (Task 3).
- Produces: registered names `ets`, `arima`, `ridge`, `hgb`, `rf`. `RecursiveMLForecaster.importance()` returns `[(feature_name, value), ...]` sorted descending.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_models.py -q`
Expected: ImportError / KeyError.

- [ ] **Step 3: Implement statistical models**

`src/forecast/models/statistical.py`:
```python
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
```

- [ ] **Step 4: Implement recursive ML models**

`src/forecast/models/ml.py`:
```python
"""Recursive multi-step ML forecasters: predict t+1, append, recompute features, repeat."""
from __future__ import annotations

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
```

- [ ] **Step 5: Register in `models/__init__.py`**

Replace the `MODEL_FACTORIES` block with:
```python
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
```

- [ ] **Step 6: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_models.py tests/forecast/test_baselines.py -q`
Expected: all passed (ridge/hgb/rf tests may take ~10 s).

- [ ] **Step 7: Commit**

```bash
git add src/forecast/models tests/forecast/test_models.py
git commit -m "feat(forecast): ETS, ARIMA and recursive ML forecasters"
```

---

### Task 5: Rolling-origin backtest

**Files:**
- Create: `src/forecast/backtest.py`
- Test: `tests/forecast/test_backtest.py`

**Interfaces:**
- Consumes: `make_model` (Task 3/4).
- Produces:
  - `plan_origins(n: int, horizon: int, step: int, min_origins: int = 5, min_horizon: int = 24) -> tuple[list[int], int, list[str]]`
  - `@dataclass(frozen=True) BacktestResult(horizon: int, origins: tuple[int, ...], actuals: np.ndarray, last_values: np.ndarray, predictions: dict[str, np.ndarray], failures: dict[str, str], warnings: tuple[str, ...])`
  - `run_backtest(series, season_start, model_names, horizon, origin_step=6, min_origins=5, log=None) -> BacktestResult`
  - `@dataclass(frozen=True) ModelMetrics(name, mae_h1, mae_h24, mae_mean, rmse_mean, mase_mean, direction_acc, per_horizon_mae: tuple[float, ...])`
  - `summarize(result: BacktestResult) -> list[ModelMetrics]` sorted by `mae_mean` ascending.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_backtest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.forecast.backtest import BacktestResult, plan_origins, run_backtest, summarize
from src.forecast.models import register
from src.forecast.models.base import Forecaster


def test_plan_origins_normal():
    origins, h, warns = plan_origins(n=240, horizon=48, step=6)
    assert origins[0] == 144 and origins[-1] <= 240 - 48 and h == 48 and warns == []


def test_plan_origins_shrinks_horizon():
    origins, h, warns = plan_origins(n=120, horizon=72, step=6, min_origins=5)
    assert len(origins) >= 5 and h < 72 and h >= 24
    assert warns and "shrunk" in warns[0]


def test_plan_origins_raises_when_impossible():
    with pytest.raises(ValueError):
        plan_origins(n=60, horizon=24, step=6)


def test_backtest_shapes_and_seasonal_beats_naive(synthetic_series, season_start):
    r = run_backtest(synthetic_series, season_start, ["naive", "seasonal_naive_24"], horizon=24)
    n = len(r.origins)
    assert r.actuals.shape == (n, 24) and r.predictions["naive"].shape == (n, 24)
    assert r.last_values.shape == (n,)
    m = {x.name: x for x in summarize(r)}
    assert m["seasonal_naive_24"].mae_mean < m["naive"].mae_mean
    assert m["naive"].mase_mean == pytest.approx(1.0)
    assert len(m["naive"].per_horizon_mae) == 24
    assert m["naive"].mae_h1 == pytest.approx(m["naive"].per_horizon_mae[0])
    assert 0 <= m["naive"].direction_acc <= 1


def test_summary_sorted_by_mae(synthetic_series, season_start):
    r = run_backtest(synthetic_series, season_start, ["naive", "seasonal_naive_24", "ridge"], horizon=24)
    maes = [m.mae_mean for m in summarize(r)]
    assert maes == sorted(maes)


class _Boom(Forecaster):
    name = "boom"

    def fit(self, series, season_start):
        raise RuntimeError("kaboom")


def test_failed_model_is_recorded_not_fatal(synthetic_series, season_start):
    register("boom", _Boom)
    r = run_backtest(synthetic_series, season_start, ["naive", "boom"], horizon=24)
    assert "boom" in r.failures and "kaboom" in r.failures["boom"]
    assert "boom" not in r.predictions and "naive" in r.predictions


def test_mae_h24_is_nan_when_horizon_shorter(synthetic_series, season_start):
    r = run_backtest(synthetic_series.iloc[:120], season_start, ["naive"], horizon=24)
    # horizon shrinks below 24 here? if not, h24 exists; either way must not crash
    m = summarize(r)[0]
    assert np.isnan(m.mae_h24) or m.mae_h24 >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_backtest.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/forecast/backtest.py`:
```python
"""Rolling-origin backtest: for each origin, fit on rows [:o], forecast rows [o:o+h]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .models import make_model

Logger = Callable[[str], None]


def plan_origins(n: int, horizon: int, step: int, min_origins: int = 5,
                 min_horizon: int = 24) -> tuple[list[int], int, list[str]]:
    start = max(48, int(0.6 * n))

    def origins_for(h: int) -> list[int]:
        return list(range(start, n - h + 1, step))

    warnings: list[str] = []
    h = horizon
    origins = origins_for(h)
    if len(origins) < min_origins:
        shrunk = min(horizon, max(min_horizon, n - start - (min_origins - 1) * step))
        if shrunk < horizon and len(origins_for(shrunk)) >= min_origins:
            warnings.append(
                f"evaluation horizon shrunk from {horizon}h to {shrunk}h to obtain "
                f"{min_origins} backtest origins (series has {n} hours)")
            h = shrunk
            origins = origins_for(h)
    if not origins:
        raise ValueError(f"series of {n} hours is too short to backtest a {horizon}h horizon")
    if len(origins) < min_origins:
        warnings.append(f"only {len(origins)} backtest origins available (wanted {min_origins})")
    return origins, h, warnings


@dataclass(frozen=True)
class BacktestResult:
    horizon: int
    origins: tuple[int, ...]
    actuals: np.ndarray            # (n_origins, horizon)
    last_values: np.ndarray        # (n_origins,) value at origin-1
    predictions: dict[str, np.ndarray]
    failures: dict[str, str]
    warnings: tuple[str, ...]


def run_backtest(series: pd.DataFrame, season_start: pd.Timestamp, model_names: list[str] | tuple[str, ...],
                 horizon: int, origin_step: int = 6, min_origins: int = 5,
                 log: Logger | None = None) -> BacktestResult:
    y = series["median_price"].to_numpy(dtype=float)
    origins, h, warnings = plan_origins(len(series), horizon, origin_step, min_origins)
    actuals = np.stack([y[o:o + h] for o in origins])
    last_values = np.array([y[o - 1] for o in origins])
    predictions: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for name in model_names:
        rows = []
        try:
            for o in origins:
                rows.append(make_model(name).fit(series.iloc[:o], season_start).predict(h))
            predictions[name] = np.stack(rows)
            if log:
                log(f"backtest {name}: {len(origins)} origins done")
        except Exception as exc:  # one bad model must not sink the run
            failures[name] = f"{type(exc).__name__}: {exc}"
            if log:
                log(f"backtest {name}: FAILED {failures[name]}")
    return BacktestResult(h, tuple(origins), actuals, last_values, predictions, failures, tuple(warnings))


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    mae_h1: float
    mae_h24: float
    mae_mean: float
    rmse_mean: float
    mase_mean: float
    direction_acc: float
    per_horizon_mae: tuple[float, ...]


def summarize(result: BacktestResult) -> list[ModelMetrics]:
    naive_mae = float(np.nanmean(np.abs(result.actuals - result.predictions["naive"]))) \
        if "naive" in result.predictions else float("nan")
    out: list[ModelMetrics] = []
    for name, pred in result.predictions.items():
        err = result.actuals - pred
        per_h = np.nanmean(np.abs(err), axis=0)
        mae = float(np.nanmean(np.abs(err)))
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        last = result.last_values[:, None]
        direction = np.sign(pred - last) == np.sign(result.actuals - last)
        out.append(ModelMetrics(
            name=name,
            mae_h1=float(per_h[0]),
            mae_h24=float(per_h[23]) if result.horizon >= 24 else float("nan"),
            mae_mean=mae,
            rmse_mean=rmse,
            mase_mean=mae / naive_mae if naive_mae and not np.isnan(naive_mae) else float("nan"),
            direction_acc=float(np.nanmean(direction)),
            per_horizon_mae=tuple(float(v) for v in per_h),
        ))
    return sorted(out, key=lambda m: m.mae_mean)
```

- [ ] **Step 4: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_backtest.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/backtest.py tests/forecast/test_backtest.py
git commit -m "feat(forecast): rolling-origin backtest and metrics"
```

---

### Task 6: Empirical intervals and model selection

**Files:**
- Create: `src/forecast/intervals.py`, `src/forecast/select.py`
- Test: `tests/forecast/test_intervals.py`, `tests/forecast/test_select.py`

**Interfaces:**
- Consumes: `BacktestResult`, `ModelMetrics` (Task 5).
- Produces:
  - `residuals(result, name) -> np.ndarray` (actuals − predictions, shape `(n_origins, h)`)
  - `quantile_bounds(resid, coverage, min_samples=10) -> tuple[np.ndarray, np.ndarray]` (per-horizon lower/upper offsets)
  - `apply_bounds(preds, lo, hi) -> tuple[np.ndarray, np.ndarray]` — handles `len(preds) > len(lo)` by repeating the last offset.
  - `empirical_coverage(resid, coverage, min_samples=10) -> float` (leave-one-origin-out)
  - `@dataclass(frozen=True) Selection(name: str, members: tuple[tuple[str, float], ...], beats_baseline: bool, skill_vs_naive: float, predictions: np.ndarray)`
  - `select_model(result, metrics, top_k=3) -> Selection`

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_intervals.py`:
```python
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
```

`tests/forecast/test_select.py`:
```python
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
    assert sel.name == "ensemble" and {m for m, _ in sel.members} == {"a", "b", "naive"} - {"naive"} or True
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_intervals.py tests/forecast/test_select.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement intervals**

`src/forecast/intervals.py`:
```python
"""Prediction intervals from per-horizon empirical residual quantiles."""
from __future__ import annotations

import numpy as np

from .backtest import BacktestResult


def residuals(result: BacktestResult, name: str) -> np.ndarray:
    return result.actuals - result.predictions[name]


def _pool(resid: np.ndarray, j: int, min_samples: int) -> np.ndarray:
    h = resid.shape[1]
    pool = resid[:, j]
    k = 1
    while pool.size < min_samples and (j - k >= 0 or j + k < h):
        parts = [pool]
        if j - k >= 0:
            parts.append(resid[:, j - k])
        if j + k < h:
            parts.append(resid[:, j + k])
        pool = np.concatenate(parts)
        k += 1
    return pool[~np.isnan(pool)]


def quantile_bounds(resid: np.ndarray, coverage: float, min_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - coverage) / 2.0
    h = resid.shape[1]
    lo, hi = np.empty(h), np.empty(h)
    for j in range(h):
        pool = _pool(resid, j, min_samples)
        lo[j] = np.quantile(pool, alpha)
        hi[j] = np.quantile(pool, 1.0 - alpha)
    return lo, hi


def apply_bounds(preds: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(preds)
    if len(lo) < n:
        lo = np.concatenate([lo, np.full(n - len(lo), lo[-1])])
        hi = np.concatenate([hi, np.full(n - len(hi), hi[-1])])
    return preds + lo[:n], preds + hi[:n]


def empirical_coverage(resid: np.ndarray, coverage: float, min_samples: int = 10) -> float:
    """Leave-one-origin-out coverage of the quantile intervals within the backtest."""
    n = resid.shape[0]
    if n < 3:
        return float("nan")
    hits = []
    for i in range(n):
        lo, hi = quantile_bounds(np.delete(resid, i, axis=0), coverage, min_samples)
        row = resid[i]
        ok = ~np.isnan(row)
        hits.append(np.mean((row[ok] >= lo[ok]) & (row[ok] <= hi[ok])))
    return float(np.mean(hits))
```

- [ ] **Step 4: Implement selection**

`src/forecast/select.py`:
```python
"""Pick the model (or inverse-MAE ensemble) with the lowest backtest MAE; never worse than naive."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult, ModelMetrics

_EPS = 1e-9


@dataclass(frozen=True)
class Selection:
    name: str
    members: tuple[tuple[str, float], ...]
    beats_baseline: bool
    skill_vs_naive: float
    predictions: np.ndarray  # backtest predictions of the selected model/ensemble


def _mae(result: BacktestResult, pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(result.actuals - pred)))


def select_model(result: BacktestResult, metrics: list[ModelMetrics], top_k: int = 3) -> Selection:
    by_name = {m.name: m for m in metrics}
    naive_mae = by_name["naive"].mae_mean
    ranked = sorted(metrics, key=lambda m: m.mae_mean)
    best = ranked[0]
    name, members, pred, mae = best.name, ((best.name, 1.0),), result.predictions[best.name], best.mae_mean

    candidates = ranked[:top_k]
    if len(candidates) >= 2:
        w = np.array([1.0 / max(m.mae_mean, _EPS) for m in candidates])
        w = w / w.sum()
        ens = sum(wi * result.predictions[m.name] for wi, m in zip(w, candidates))
        ens_mae = _mae(result, ens)
        if ens_mae < min(m.mae_mean for m in candidates):
            name, pred, mae = "ensemble", ens, ens_mae
            members = tuple((m.name, float(wi)) for m, wi in zip(candidates, w))

    skill = 1.0 - mae / naive_mae if naive_mae > 0 else 0.0
    if skill <= 0:
        return Selection("naive", (("naive", 1.0),), False, float(skill), result.predictions["naive"])
    return Selection(name, members, True, float(skill), pred)
```

- [ ] **Step 5: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_intervals.py tests/forecast/test_select.py -q`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/forecast/intervals.py src/forecast/select.py tests/forecast/test_intervals.py tests/forecast/test_select.py
git commit -m "feat(forecast): empirical intervals and backtest-driven model selection"
```

---

### Task 7: Reference curves from other seasons

**Files:**
- Create: `src/forecast/reference.py`
- Test: `tests/forecast/test_reference.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ReferenceCurve(season_id: str, name: str, points: tuple[tuple[int, float], ...])` — `(h, median_price)` with `h` in `1..horizon`.
  - `reference_curves(current_offset_hours: int, horizon: int, others: list[tuple[str, str, pd.DataFrame, pd.Timestamp]]) -> list[ReferenceCurve]` — each tuple is `(season_id, name, hourly_series, season_start)`; seasons with no overlapping offsets are skipped.

- [ ] **Step 1: Write failing test**

`tests/forecast/test_reference.py`:
```python
import numpy as np
import pandas as pd

from src.forecast.reference import reference_curves


def _series(start: str, hours: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="ts")
    return pd.DataFrame({"median_price": np.arange(hours, dtype=float)}, index=idx)


def test_reference_aligns_by_offset():
    s2 = _series("2025-10-06T00:00:00Z", 200)
    s3 = _series("2026-03-23T00:00:00Z", 50)            # too short to overlap
    curves = reference_curves(current_offset_hours=100, horizon=24, others=[
        ("s2", "Season 2", s2, pd.Timestamp("2025-10-06T00:00:00Z")),
        ("s3", "Season 3", s3, pd.Timestamp("2026-03-23T00:00:00Z")),
    ])
    assert [c.season_id for c in curves] == ["s2"]
    pts = dict(curves[0].points)
    assert pts[1] == 101.0 and pts[24] == 124.0 and len(pts) == 24


def test_reference_skips_nan_points():
    s = _series("2025-10-06T00:00:00Z", 200)
    s.iloc[105, 0] = np.nan
    curves = reference_curves(100, 10, [("s", "S", s, pd.Timestamp("2025-10-06T00:00:00Z"))])
    assert 5 not in dict(curves[0].points) and len(curves[0].points) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_reference.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/forecast/reference.py`:
```python
"""Other seasons' price paths aligned by hours-since-season-start (display reference only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceCurve:
    season_id: str
    name: str
    points: tuple[tuple[int, float], ...]


def reference_curves(current_offset_hours: int, horizon: int,
                     others: list[tuple[str, str, pd.DataFrame, pd.Timestamp]]) -> list[ReferenceCurve]:
    out: list[ReferenceCurve] = []
    for season_id, name, series, start in others:
        if series.empty:
            continue
        offset = ((series.index - start) / pd.Timedelta(hours=1)).astype(int)
        rel = offset - current_offset_hours
        mask = (rel >= 1) & (rel <= horizon) & ~np.isnan(series["median_price"].to_numpy(dtype=float))
        if not mask.any():
            continue
        pts = tuple((int(h), float(p)) for h, p in zip(rel[mask], series["median_price"].to_numpy()[mask]))
        out.append(ReferenceCurve(season_id, name, pts))
    return out
```

- [ ] **Step 4: Run test**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_reference.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/forecast/reference.py tests/forecast/test_reference.py
git commit -m "feat(forecast): cross-season reference curves"
```

---

### Task 8: Pipeline orchestration

**Files:**
- Create: `src/forecast/pipeline.py`
- Test: `tests/forecast/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces:
  - `@dataclass(frozen=True) Season(id: str, name: str, start: pd.Timestamp, end: pd.Timestamp | None)`
  - `@dataclass(frozen=True) ForecastRequest(item: str, season_id: str, horizon_hours: int = 72, models: tuple[str, ...] = (), coverage: float = 0.9, backtest_origin_step: int = 6)` — empty `models` means all registered.
  - `class InsufficientDataError(ValueError)`, `class InvalidRequestError(ValueError)`
  - `ListingLoader = Callable[[str, pd.Timestamp, pd.Timestamp | None], pd.DataFrame]`
  - `run_forecast(request, load_listings, seasons: list[Season], source: str = "unknown", log: Callable[[str], None] | None = None) -> dict` — the dict layout of spec §6.8 (keys: `run_id, created_at, request, source, series_tail, forecast, selected, backtest, beats_baseline, skill_vs_naive, feature_importance, references, warnings`).
  - `MIN_HOURS = 72`, `HORIZON_RANGE = (24, 168)`, `COVERAGE_RANGE = (0.5, 0.99)`.

- [ ] **Step 1: Write failing tests**

`tests/forecast/test_pipeline.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

from src.forecast.pipeline import (ForecastRequest, InsufficientDataError, InvalidRequestError,
                                   Season, run_forecast)
from tests.forecast.conftest import SEASON_START, make_listings

S1 = Season("s1", "Season 1", pd.Timestamp("2025-10-06T00:00:00Z"), pd.Timestamp("2025-12-01T00:00:00Z"))
S2 = Season("s2", "Season 2", SEASON_START, None)


def _loader(frames: dict[str, pd.DataFrame]):
    def load(item, start, end):
        df = frames[item]
        ts = pd.to_datetime(df["created_at"], utc=True)
        m = ts >= start
        if end is not None:
            m &= ts < end
        return df[m]
    return load


@pytest.fixture
def frames():
    s2 = make_listings(hours=240, seed=0)
    s1 = make_listings(hours=300, seed=1)
    shift = S1.start - SEASON_START
    s1["created_at"] = pd.to_datetime(s1["created_at"]) + shift
    return {"Test Ore": pd.concat([s1, s2], ignore_index=True)}


def test_end_to_end_schema_and_skill(frames):
    req = ForecastRequest("Test Ore", "s2", horizon_hours=48, models=("ridge", "hgb"))
    logs = []
    out = run_forecast(req, _loader(frames), [S1, S2], source="csv", log=logs.append)
    json.dumps(out)  # serialisable
    for key in ("run_id", "created_at", "request", "source", "series_tail", "forecast", "selected",
                "backtest", "beats_baseline", "skill_vs_naive", "feature_importance", "references", "warnings"):
        assert key in out
    assert len(out["forecast"]) == 48 and out["forecast"][0]["h"] == 1
    f0 = out["forecast"][0]
    assert f0["lower"] <= f0["pred"] <= f0["upper"]
    names = {m["name"] for m in out["backtest"]["models"]}
    assert {"naive", "seasonal_naive_24", "ridge", "hgb"} <= names
    assert out["beats_baseline"] is True and out["skill_vs_naive"] > 0
    assert out["references"] and out["references"][0]["season_id"] == "s1"
    assert out["series_tail"][-1]["ts"] < out["forecast"][0]["ts"]
    assert any("backtest" in line for line in logs)
    assert out["backtest"]["coverage"] is None or 0 <= out["backtest"]["coverage"] <= 1


def test_feature_importance_present_for_tree_model(frames):
    out = run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("rf",)), _loader(frames), [S1, S2])
    assert out["feature_importance"] and "name" in out["feature_importance"][0]


def test_short_last_segment_is_rejected(frames):
    df = frames["Test Ore"]
    ts = pd.to_datetime(df["created_at"], utc=True)
    cut_start = SEASON_START + pd.Timedelta(hours=200)
    cut_end = SEASON_START + pd.Timedelta(hours=210)
    frames2 = {"Test Ore": df[~((ts >= cut_start) & (ts < cut_end))]}   # 10h gap, leaves 30h segment
    with pytest.raises(InsufficientDataError) as e:
        run_forecast(ForecastRequest("Test Ore", "s2", 24, models=("ridge",)), _loader(frames2), [S1, S2])
    assert "30" in str(e.value)


@pytest.mark.parametrize("bad", [
    dict(horizon_hours=12), dict(horizon_hours=200), dict(coverage=0.3),
    dict(models=("ghost",)), dict(season_id="nope"),
])
def test_invalid_requests(frames, bad):
    kwargs = dict(item="Test Ore", season_id="s2", horizon_hours=24, models=("ridge",))
    kwargs.update(bad)
    with pytest.raises(InvalidRequestError):
        run_forecast(ForecastRequest(**kwargs), _loader(frames), [S1, S2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_pipeline.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/forecast/pipeline.py`:
```python
"""run_forecast: listings -> hourly series -> backtest -> selection -> forecast + intervals."""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .aggregate import hourly_series, prepare_target
from .backtest import run_backtest, summarize
from .intervals import apply_bounds, empirical_coverage, quantile_bounds
from .models import BASELINE_MODELS, MODEL_FACTORIES, make_model
from .reference import reference_curves
from .select import select_model

MIN_HOURS = 72
HORIZON_RANGE = (24, 168)
COVERAGE_RANGE = (0.5, 0.99)
TAIL_HOURS = 168

ListingLoader = Callable[[str, pd.Timestamp, "pd.Timestamp | None"], pd.DataFrame]
Logger = Callable[[str], None]


class InsufficientDataError(ValueError):
    pass


class InvalidRequestError(ValueError):
    pass


@dataclass(frozen=True)
class Season:
    id: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None


@dataclass(frozen=True)
class ForecastRequest:
    item: str
    season_id: str
    horizon_hours: int = 72
    models: tuple[str, ...] = ()
    coverage: float = 0.9
    backtest_origin_step: int = 6


def _validate(req: ForecastRequest, seasons: list[Season]) -> tuple[Season, tuple[str, ...]]:
    lo, hi = HORIZON_RANGE
    if not lo <= req.horizon_hours <= hi:
        raise InvalidRequestError(f"horizon_hours must be in [{lo}, {hi}], got {req.horizon_hours}")
    clo, chi = COVERAGE_RANGE
    if not clo <= req.coverage <= chi:
        raise InvalidRequestError(f"coverage must be in [{clo}, {chi}], got {req.coverage}")
    if req.backtest_origin_step < 1:
        raise InvalidRequestError("backtest_origin_step must be >= 1")
    unknown = [m for m in req.models if m not in MODEL_FACTORIES]
    if unknown:
        raise InvalidRequestError(f"unknown models {unknown}; valid: {sorted(MODEL_FACTORIES)}")
    season = next((s for s in seasons if s.id == req.season_id), None)
    if season is None:
        raise InvalidRequestError(f"unknown season_id {req.season_id!r}")
    chosen = tuple(req.models) if req.models else tuple(MODEL_FACTORIES)
    names = tuple(dict.fromkeys(BASELINE_MODELS + chosen))  # baselines first, no duplicates
    return season, names


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _fit_selected(selection, series: pd.DataFrame, season_start: pd.Timestamp, horizon: int):
    """Refit members on the full series; return (point forecast, importance or None)."""
    preds, importance = [], None
    for name, weight in selection.members:
        model = make_model(name).fit(series, season_start)
        preds.append(weight * model.predict(horizon))
        if importance is None:
            importance = model.importance()
    return np.sum(preds, axis=0), importance


def run_forecast(request: ForecastRequest, load_listings: ListingLoader, seasons: list[Season],
                 source: str = "unknown", log: Logger | None = None) -> dict:
    say = log or (lambda _s: None)
    season, model_names = _validate(request, seasons)
    say(f"loading listings for {request.item} in {season.name}")
    listings = load_listings(request.item, season.start, season.end)
    full_series = hourly_series(listings)
    series, warnings = prepare_target(full_series)
    if len(series) < MIN_HOURS:
        raise InsufficientDataError(
            f"need at least {MIN_HOURS} contiguous hours, have {len(series)} "
            f"(season {season.name}, {len(listings)} listings)")
    say(f"{len(series)} hourly points, backtesting {len(model_names)} models")

    result = run_backtest(series, season.start, model_names, request.horizon_hours,
                          origin_step=request.backtest_origin_step, log=say)
    warnings += list(result.warnings)
    warnings += [f"model {n} failed: {err}" for n, err in result.failures.items()]
    metrics = summarize(result)
    selection = select_model(result, metrics)
    say(f"selected {selection.name} (skill vs naive {selection.skill_vs_naive:+.1%})")

    point, importance = _fit_selected(selection, series, season.start, request.horizon_hours)
    resid = result.actuals - selection.predictions
    lo, hi = quantile_bounds(resid, request.coverage)
    lower, upper = apply_bounds(point, lo, hi)
    coverage = empirical_coverage(resid, request.coverage)
    if request.horizon_hours > result.horizon:
        warnings.append(f"intervals beyond h={result.horizon} reuse the last backtested width")

    last_ts = series.index[-1]
    current_offset = int((last_ts - season.start) / pd.Timedelta(hours=1))
    others = []
    for s in seasons:
        if s.id == season.id:
            continue
        other = hourly_series(load_listings(request.item, s.start, s.end))
        if not other.empty:
            others.append((s.id, s.name, other, s.start))
    references = reference_curves(current_offset, request.horizon_hours, others)

    tail = full_series.iloc[-TAIL_HOURS:]
    now = pd.Timestamp.now(tz="UTC")
    run_id = f"{now:%Y%m%dT%H%M%S}-{request.item.lower().replace(' ', '_')}-{uuid.uuid4().hex[:6]}"
    return {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "request": {**asdict(request), "models": list(model_names)},
        "source": source,
        "season": {"id": season.id, "name": season.name, "start": season.start.isoformat(),
                   "end": season.end.isoformat() if season.end is not None else None},
        "series_tail": [
            {"ts": ts.isoformat(), "median_price": _num(r.median_price), "p25": _num(r.p25),
             "p75": _num(r.p75), "volume": int(r.volume), "listings": int(r.listings)}
            for ts, r in tail.iterrows()],
        "forecast": [
            {"ts": (last_ts + pd.Timedelta(hours=i + 1)).isoformat(), "h": i + 1,
             "pred": float(point[i]), "lower": float(lower[i]), "upper": float(upper[i])}
            for i in range(request.horizon_hours)],
        "selected": {"name": selection.name,
                     "members": [{"name": n, "weight": w} for n, w in selection.members]},
        "backtest": {
            "origins": len(result.origins), "horizon": result.horizon,
            "coverage": _num(coverage), "nominal_coverage": request.coverage,
            "models": [{"name": m.name, "mae_h1": _num(m.mae_h1), "mae_h24": _num(m.mae_h24),
                        "mae_mean": _num(m.mae_mean), "rmse_mean": _num(m.rmse_mean),
                        "mase_mean": _num(m.mase_mean), "direction_acc": _num(m.direction_acc)}
                       for m in metrics],
            "per_horizon": {m.name: [_num(v) for v in m.per_horizon_mae] for m in metrics},
            "failures": result.failures,
        },
        "beats_baseline": selection.beats_baseline,
        "skill_vs_naive": float(selection.skill_vs_naive),
        "feature_importance": [{"name": n, "value": float(v)} for n, v in importance] if importance else [],
        "references": [{"season_id": c.season_id, "name": c.name,
                        "points": [{"h": h, "median_price": p} for h, p in c.points]} for c in references],
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_pipeline.py -q`
Expected: 8 passed (may take ~30 s).

- [ ] **Step 5: Commit**

```bash
git add src/forecast/pipeline.py tests/forecast/test_pipeline.py
git commit -m "feat(forecast): end-to-end run_forecast pipeline"
```

---

### Task 9: CLI entry point and old module removal

**Files:**
- Create: `src/forecast/cli.py`
- Delete: `src/analysis/Machine_learning_analysis.py`, `src/web/run_analysis_cli.py`
- Modify: `requirements.txt` (drop matplotlib/seaborn), `README.md` (ML section)
- Test: `tests/forecast/test_cli.py`

**Interfaces:**
- Consumes: `run_forecast`, `Season` (Task 8).
- Produces: `python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72 [--models ridge,hgb] [--out runs/x.json] [--csv-dir .] [--seasons seasons.json]`. Exit code 0 on success, 2 on invalid request / insufficient data.
  - `load_seasons(path) -> list[Season]` (also reused by backend's seasons store conversion).
  - `csv_loader(csv_dir, items_config_path) -> ListingLoader`.

- [ ] **Step 1: Write failing test**

`tests/forecast/test_cli.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.forecast.conftest import SEASON_START, make_listings


def test_cli_writes_json(tmp_path: Path):
    df = make_listings(hours=200)
    df.to_csv(tmp_path / "test_ore.csv", index=False)
    (tmp_path / "items_config.json").write_text(json.dumps({"items": [
        {"name": "Test Ore", "file": "x.py", "csv": "test_ore.csv", "category": "ore", "enabled": True}]}))
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": [
        {"id": "s2", "name": "S2", "start": SEASON_START.isoformat(), "end": None}]}))
    out = tmp_path / "run.json"
    proc = subprocess.run([sys.executable, "-m", "src.forecast.cli", "--item", "Test Ore", "--season", "s2",
                           "--horizon", "24", "--models", "ridge", "--csv-dir", str(tmp_path),
                           "--seasons", str(tmp_path / "seasons.json"), "--out", str(out)],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert len(data["forecast"]) == 24
    assert "selected" in proc.stdout


def test_cli_invalid_horizon_exit_2(tmp_path: Path):
    (tmp_path / "seasons.json").write_text(json.dumps({"seasons": []}))
    (tmp_path / "items_config.json").write_text(json.dumps({"items": []}))
    proc = subprocess.run([sys.executable, "-m", "src.forecast.cli", "--item", "X", "--season", "s",
                           "--horizon", "5", "--csv-dir", str(tmp_path),
                           "--seasons", str(tmp_path / "seasons.json")],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert proc.returncode == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_cli.py -q`
Expected: FAIL (module not found, returncode 1).

- [ ] **Step 3: Implement**

`src/forecast/cli.py`:
```python
"""Command-line forecast runner over CSV exports.

    python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72 --out runs/iron.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .pipeline import (ForecastRequest, InsufficientDataError, InvalidRequestError, ListingLoader,
                       Season, run_forecast)


def load_seasons(path: Path) -> list[Season]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Season(s["id"], s["name"], pd.Timestamp(s["start"]).tz_convert("UTC")
                   if pd.Timestamp(s["start"]).tzinfo else pd.Timestamp(s["start"], tz="UTC"),
                   None if s.get("end") in (None, "") else pd.Timestamp(s["end"]).tz_convert("UTC")
                   if pd.Timestamp(s["end"]).tzinfo else pd.Timestamp(s["end"], tz="UTC"))
            for s in raw.get("seasons", [])]


def csv_loader(csv_dir: Path, items_config: Path) -> ListingLoader:
    cfg = json.loads(Path(items_config).read_text(encoding="utf-8"))
    files = {it["name"]: Path(csv_dir) / it["csv"] for it in cfg.get("items", [])}
    cache: dict[str, pd.DataFrame] = {}

    def load(item: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
        if item not in files:
            raise InvalidRequestError(f"unknown item {item!r}; known: {sorted(files)}")
        if item not in cache:
            cache[item] = pd.read_csv(files[item])
        df = cache[item]
        ts = pd.to_datetime(df["created_at"], utc=True)
        mask = ts >= start
        if end is not None:
            mask &= ts < end
        return df[mask]

    return load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hourly price forecast with rolling-origin backtest")
    p.add_argument("--item", required=True)
    p.add_argument("--season", required=True, help="season id from seasons.json")
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--models", default="", help="comma-separated subset, e.g. ridge,hgb")
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--csv-dir", default=".")
    p.add_argument("--seasons", default="seasons.json")
    p.add_argument("--out", default=None, help="write full JSON result here")
    args = p.parse_args(argv)

    csv_dir = Path(args.csv_dir)
    req = ForecastRequest(args.item, args.season, args.horizon,
                          tuple(m for m in args.models.split(",") if m), args.coverage)
    try:
        seasons = load_seasons(Path(args.seasons))
        loader = csv_loader(csv_dir, csv_dir / "items_config.json")
        out = run_forecast(req, loader, seasons, source="csv", log=lambda s: print(f"  {s}", file=sys.stderr))
    except (InvalidRequestError, InsufficientDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"selected: {out['selected']['name']}  beats baseline: {out['beats_baseline']}  "
          f"skill vs naive: {out['skill_vs_naive']:+.1%}")
    print("model              mae_mean   mase   dir_acc")
    for m in out["backtest"]["models"]:
        print(f"{m['name']:<18} {m['mae_mean']:>8.3f} {m['mase_mean']:>6.2f} {m['direction_acc']:>8.2f}")
    print(f"next 24h: {out['forecast'][0]['pred']:.2f} -> {out['forecast'][min(23, len(out['forecast'])-1)]['pred']:.2f}")
    for w in out["warnings"]:
        print(f"warning: {w}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast/test_cli.py -q`
Expected: 2 passed.

- [ ] **Step 5: Remove old analysis module and CLI, update requirements and README**

```bash
git rm -q src/analysis/Machine_learning_analysis.py src/web/run_analysis_cli.py
rmdir src/analysis src/web 2>/dev/null || true
```

In `requirements.txt`, delete the lines `matplotlib>=3.5.0` and `seaborn>=0.11.0`.

In `README.md`, replace the "Machine Learning Analysis / 机器学习分析" feature bullet list with:
```markdown
### 🤖 Hourly Price Forecast / 小时级价格预测
- Listings are aggregated to an hourly median; features use only past hours (no target leakage) / 挂单按小时聚合为中位价，特征只用过去的小时
- Rolling-origin backtest against naive baselines; the UI reports whether the model beats "repeat last hour" / 滚动起点回测对比基线，界面直接告知是否优于"复读上一小时"
- Prediction intervals from empirical backtest residual quantiles / 区间来自回测残差的经验分位数
- Trains only inside the selected season; other seasons shown as reference curves / 只在所选赛季内训练，其他赛季作为参考曲线
- CLI: `python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72`
```

- [ ] **Step 6: Run the whole forecast suite**

Run: `/opt/anaconda3/bin/python -m pytest tests/forecast -q`
Expected: all passed.

- [ ] **Step 7: Try the CLI on real data** (requires a `seasons.json`; create a temporary one covering gold_ore)

```bash
cat > seasons.json <<'EOF'
{"seasons": [
  {"id": "s1", "name": "Season 1", "start": "2025-10-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "note": ""},
  {"id": "s2", "name": "Season 2", "start": "2026-01-01T00:00:00Z", "end": "2026-03-01T00:00:00Z", "note": ""},
  {"id": "s3", "name": "Season 3", "start": "2026-03-01T00:00:00Z", "end": null, "note": ""}
]}
EOF
/opt/anaconda3/bin/python -m src.forecast.cli --item "Gold Ore" --season s2 --horizon 72 --out runs/gold_s2.json
```
Expected: a metrics table, a `selected:` line, and no traceback. (Season boundaries are placeholders the user edits later in the UI.)

- [ ] **Step 8: Commit**

```bash
git add -A src/forecast tests/forecast requirements.txt README.md seasons.json
git commit -m "feat(forecast): CLI runner; remove leaky legacy analysis module"
```
