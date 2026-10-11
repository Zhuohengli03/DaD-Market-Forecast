from fastapi import APIRouter, HTTPException, Request

from ..schemas import RunTaskIn, ok

router = APIRouter(tags=["scheduler"])


@router.get("/scheduler")
def scheduler_status(request: Request):
    sc = request.app.state.scheduler
    return ok({"running": sc.running(), "tasks": sc.tasks()})


@router.post("/scheduler/start")
def scheduler_start(request: Request):
    st = request.app.state
    try:
        running = st.scheduler.start()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    st.app_settings.update(auto_collect=True)        # remembered: resumes when the app starts again
    return ok({"running": running})


@router.post("/scheduler/stop")
def scheduler_stop(request: Request):
    st = request.app.state
    st.app_settings.update(auto_collect=False)
    return ok({"running": st.scheduler.stop()})


@router.post("/scheduler/run", status_code=202)
def scheduler_run(body: RunTaskIn, request: Request):
    try:
        job = request.app.state.scheduler.run_task(body.script_path)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, f"script not found: {exc}")
    return ok({"job_id": job.id})
