# API Reference / 接口参考

Base URL: `http://localhost:5001/api`. Every response uses one envelope:

```json
{ "ok": true, "data": ..., "error": null, "meta": null }
```

Validation failures return HTTP 422 with `error` listing the offending fields. Timestamps are ISO-8601 UTC.

| Method | Path | Description |
|---|---|---|
| GET | `/status` | data source (`postgres`/`csv`), per-item row counts and latest timestamps, running jobs, scheduler state |
| GET | `/setup` | first-run state: `api_key_configured`, `api_key_hint`, `data_source`, counts of seasons, tracked items, catalog items |
| PUT | `/setup/api-key` | `{api_key}`: verified against DarkerDB, then saved to `.env`; 422 if DarkerDB rejects it, 502 if unreachable |
| GET | `/catalog` | query `q` / `rarity` / `item_type` / `tradable` / `page` / `page_size` (≤200); rows carry summary fields, `market` stats and the full `raw` DarkerDB record; `meta.facets` lists filter values |
| GET | `/catalog/meta` | patch, last sync, last market scan, counts |
| POST | `/catalog/sync` | job: download the whole `/v2/items` catalog → 202 `{job_id}`; 409 without an API key |
| POST | `/catalog/scan-market` | `{max_pages}` (default 40): job reading newest `/v2/market` pages since the latest season start |
| GET | `/items` | tracked items (`data/items.json`) merged with stored-data stats |
| POST | `/items/track` | `{item_id}` from the synced catalog → 201; 409 if already tracked, 404 if not in the catalog |
| DELETE | `/items/{name}` | stop tracking (collected data is kept) |
| GET | `/series/{item}` | query `season_id` / `start` / `end`, `resolution=1h\|4h\|1d`; hourly aggregated series |
| GET | `/listings/{item}` | query `season_id` / `start` / `end` / `sold` / `min_qty` / `max_qty` / `page` / `page_size` (≤500); paginated raw listings, `meta.total` |
| GET | `/seasons` | list seasons |
| POST | `/seasons` | add `{name, start, end, note}`; 422 on overlap or end ≤ start |
| PUT | `/seasons/{id}` | update |
| DELETE | `/seasons/{id}` | delete |
| GET | `/forecast/models` | registered model names |
| POST | `/forecast` | `{item, season_id, horizon_hours 24–168, models[], coverage 0.5–0.99, backtest_origin_step 1–24}` → 202 `{job_id}` |
| GET | `/forecast/runs` | run summaries, newest first |
| GET | `/forecast/runs/{run_id}` | full result (series tail, forecast with intervals, backtest, selection, references, warnings) |
| GET | `/forecast/runs/{run_id}/actuals` | hourly medians observed after the run, aligned to its forecast timestamps |
| DELETE | `/forecast/runs/{run_id}` | delete a run |
| GET | `/jobs` | all jobs |
| GET | `/jobs/{id}` | job status, error, result reference, last 200 log lines |
| GET | `/jobs/{id}/events` | Server-Sent Events: `log` lines and `status` changes; closes on completion |
| POST | `/collect/{item}` | collect a tracked item → 202 `{job_id}`; 404 if not tracked, 409 if already running |
| GET | `/scheduler` | scheduler running flag and one daily task per enabled tracked item |
| POST | `/scheduler/start`, `/scheduler/stop` | control the scheduler subprocess |
| POST | `/scheduler/run` | `{script_path}` (inside the project) → 202 `{job_id}` |
| GET | `/config/items` | `items_config.json` |
| PUT | `/config/items` | replace `items_config.json` after schema validation |

CLI equivalent for forecasts: `python -m src.forecast.cli --item "Iron Ore" --season s3 --horizon 72`.

Server errors return a reference id (`ref xxxx`) instead of internal details; the full traceback is in the backend log.
