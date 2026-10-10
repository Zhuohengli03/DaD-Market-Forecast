from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services.collection import start_collection

router = APIRouter(tags=["collect"])


@router.post("/collect/{item}", status_code=202)
def collect(item: str, request: Request):
    try:
        job = start_collection(request.app.state, item)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return ok({"job_id": job.id})


@router.post("/collect-all", status_code=202)
def collect_all(request: Request):
    from ..services.initializer import run_collect_all
    st = request.app.state
    if not st.items.list(enabled_only=True):
        raise HTTPException(422, "还没有追踪任何道具 / no tracked items")
    if st.jobs.find_active("collect"):
        raise HTTPException(409, "已有采集正在进行 / a collection is already running")
    from ..services.cost import collect_all_estimate
    job = st.jobs.create("collect", "采集全部道具 / collect all items", meta={"item": "*"})
    st.jobs.set_estimate(job, collect_all_estimate(st)["requests"])
    st.jobs.run_in_thread(job, lambda j: run_collect_all(st, j))
    return ok({"job_id": job.id})
