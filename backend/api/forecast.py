from fastapi import APIRouter, HTTPException, Request

from src.forecast.models import MODEL_FACTORIES
from src.forecast.pipeline import ForecastRequest, run_forecast
from ..schemas import ForecastIn, ok

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
    req = ForecastRequest(body.item, body.season_id, body.horizon_hours, tuple(body.models), body.coverage,
                          body.backtest_origin_step)
    job = st.jobs.create("forecast", f"{body.item} · {season['name']} · {body.horizon_hours}h",
                         meta={"item": body.item, "season_id": body.season_id})
    repo, seasons, runs, jm = st.repo, st.seasons, st.runs, st.jobs

    def work(j):
        result = run_forecast(req, repo.fetch_all, seasons.to_pipeline(), source=repo.source_name(),
                              log=lambda line: jm.append_log(j, line))
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
