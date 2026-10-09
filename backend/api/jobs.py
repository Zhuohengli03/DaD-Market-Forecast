from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..schemas import ok

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def list_jobs(request: Request):
    return ok([j.to_dict() for j in request.app.state.jobs.list()])


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    try:
        return ok(request.app.state.jobs.get(job_id).to_dict())
    except KeyError:
        raise HTTPException(404, f"job {job_id} not found")


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    jm = request.app.state.jobs
    try:
        jm.get(job_id)
    except KeyError:
        raise HTTPException(404, f"job {job_id} not found")
    return EventSourceResponse(jm.events(job_id))
