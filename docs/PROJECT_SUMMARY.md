# Architecture / 架构

```
DarkerDB v2 API ──> src/api (client, collector, CSV/Postgres storage)
                         │
                         ▼
        data/*.csv or Postgres ──> backend (FastAPI) ──> frontend (React SPA)
                                        │
                                        └──> src/forecast (pipeline)
```

| Part | Path | Notes |
|---|---|---|
| API client | `src/api/darkerdb_client.py` | `X-API-Key`, cursor paging, 60 req/min throttle, `Retry-After` on 429 |
| Collector | `src/api/collect.py`, `src/api/storage.py` | de-duplicates by listing id + timestamp; CSV unless Postgres is configured |
| Forecast | `src/forecast/` | hourly median, lag-only features, rolling-origin backtest, empirical intervals, per-season training |
| Backend | `backend/` | REST + Server-Sent Events for live job logs; user state in `data/` |
| Frontend | `frontend/` | React, TanStack Query, ECharts, zh/en |
| Scheduler | `src/scheduler/` | daily collection per tracked item |

Design notes: `docs/superpowers/specs/`.
