from fastapi import APIRouter, HTTPException, Request

from src.forecast.models import MODEL_FACTORIES
from src.forecast.holdout import HoldoutRequest, run_season_holdout
from src.forecast.pipeline import ForecastRequest, run_forecast
from ..schemas import ForecastIn, HoldoutIn, ok

router = APIRouter(tags=["forecast"])


@router.get("/forecast/models")
def models():
    return ok(list(MODEL_FACTORIES))


@router.post("/forecast", status_code=202)
def start_forecast(body: ForecastIn, request: Request):
    st = request.app.state
    unknown = [m for m in body.models if m not in MODEL_FACTORIES]
    if unknown:
        raise HTTPException(422, f"unknown models {unknown}")
    try:
        season = st.seasons.get(body.season_id)
    except KeyError:
        raise HTTPException(404, f"season {body.season_id} not found")
    if body.horizon_hours % {"1h": 1, "4h": 4, "1d": 24}[body.resolution]:
        raise HTTPException(422, "预测时长必须是时间粒度的整数倍 / horizon must be a whole number of blocks")
    req = ForecastRequest(body.item, body.season_id, body.horizon_hours, tuple(body.models), body.coverage,
                          body.backtest_origin_step, body.resolution)
    job = st.jobs.create("forecast", f"{body.item} · {season['name']} · {body.horizon_hours}h",
                         meta={"item": body.item, "season_id": body.season_id})
    repo, seasons, runs, jm = st.repo, st.seasons, st.runs, st.jobs

    def work(j):
        result = run_forecast(req, repo.fetch_all, seasons.to_pipeline(), source=repo.source_name(),
                              log=lambda line: jm.append_log(j, line),
                              progress=lambda stage, done=None, total=None, **d: jm.progress(j, stage, done, total, detail=d))
        return runs.save(result)

    jm.run_in_thread(job, work)
    return ok({"job_id": job.id})


@router.get("/forecast/runs")
def list_runs(request: Request):
    return ok(request.app.state.runs.list())


@router.get("/forecast/runs/{run_id}")
def get_run(run_id: str, request: Request):
    try:
        return ok(request.app.state.runs.get(run_id))
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")


@router.get("/forecast/runs/{run_id}/actuals")
def run_actuals(run_id: str, request: Request):
    st = request.app.state
    try:
        run = st.runs.get(run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")
    return ok(st.runs.actuals(run, st.repo))


@router.delete("/forecast/runs/{run_id}")
def delete_run(run_id: str, request: Request):
    try:
        request.app.state.runs.delete(run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")
    return ok({"deleted": run_id})


@router.post("/forecast/holdout", status_code=202)
def start_holdout(body: HoldoutIn, request: Request):
    """Season holdout: pick the method on the first part of a season, score it day by day on the rest."""
    st = request.app.state
    unknown = [m for m in body.models if m not in MODEL_FACTORIES]
    if unknown:
        raise HTTPException(422, f"unknown models {unknown}")
    try:
        season = st.seasons.get(body.season_id)
    except KeyError:
        raise HTTPException(404, f"season {body.season_id} not found")
    req = HoldoutRequest(body.item, body.season_id, body.train_frac, body.horizon_hours, tuple(body.models), body.coverage,
                         body.resolution)
    job = st.jobs.create("forecast", f"回测 {body.item} · {season['name']} / holdout",
                         meta={"item": body.item, "season_id": body.season_id, "holdout": True})
    repo, seasons, store, jm = st.repo, st.seasons, st.holdouts, st.jobs

    def work(j):
        result = run_season_holdout(req, repo.fetch_all, seasons.to_pipeline(), log=lambda line: jm.append_log(j, line),
                                    progress=lambda stage, done=None, total=None, **d: jm.progress(j, stage, done, total, detail=d))
        return store.save(result)

    jm.run_in_thread(job, work)
    return ok({"job_id": job.id})


def _holdout_summary(run: dict) -> dict:
    m = run["metrics"]
    return {"run_id": run["run_id"], "created_at": run["created_at"], "item": run["item"],
            "season_id": run["season"]["id"], "season_name": run["season"]["name"],
            "train_frac": run["request"]["train_frac"], "selected": run["selected"]["name"],
            "skill": m["skill"], "mae": m["mae"], "naive_mae": m["naive_mae"]}


@router.get("/forecast/holdouts")
def list_holdouts(request: Request):
    import json
    out = []
    for p in request.app.state.settings.runs_dir.joinpath("holdout").glob("*.json"):
        try:
            out.append(_holdout_summary(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return ok(sorted(out, key=lambda r: r["created_at"], reverse=True))


@router.get("/forecast/holdouts/{run_id}")
def get_holdout(run_id: str, request: Request):
    try:
        return ok(request.app.state.holdouts.get(run_id))
    except KeyError:
        raise HTTPException(404, f"holdout {run_id} not found")


@router.delete("/forecast/holdouts/{run_id}")
def delete_holdout(run_id: str, request: Request):
    try:
        request.app.state.holdouts.delete(run_id)
    except KeyError:
        raise HTTPException(404, f"holdout {run_id} not found")
    return ok({"deleted": run_id})


@router.get("/next-hour/{item}")
def next_hour(item: str, request: Request, season_id: str | None = None):
    """Where next hour's sales will land (P10..P90), from this machine's data only (no DarkerDB credits)."""
    import pandas as pd
    from src.forecast.next_hour import CALIB_DAYS, CHECK_DAYS, next_hour_range
    from src.forecast.pipeline import InsufficientDataError
    from src.season_calendar import market_start
    from ..services.collection import market_not_open, pick_season
    from .series import item_param
    st = request.app.state
    item = item_param(item)
    try:
        season = pick_season(st, season_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'"))
    if season is None:
        raise HTTPException(409, "还没有赛季，先在设置里导入赛季 / no seasons yet")
    opens = market_not_open(st, season["id"])
    if opens:
        raise HTTPException(409, f"这个赛季的交易行 {opens} 才开放，开放后有了成交才能估计价格范围 / the market opens on {opens}")
    start, end = pd.Timestamp(market_start(season)), pd.Timestamp(season["end"]) if season.get("end") else None
    try:
        last = next((i.last_ts for i in st.item_stats.get() if i.name == item), None)
        if last:   # only the days the range and its check need, counted back from the season's own last data
            anchor = min(pd.Timestamp(last), end) if end is not None else pd.Timestamp(last)
            start = max(start, anchor - pd.Timedelta(days=CALIB_DAYS + CHECK_DAYS + 1))
        listings = st.repo.fetch_all(item, start, end)
        return ok(next_hour_range(listings))
    except KeyError:
        raise HTTPException(404, f"没有这个道具的数据 / unknown item {item}")
    except InsufficientDataError as exc:
        raise HTTPException(409, str(exc))
