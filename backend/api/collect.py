from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services.collection import start_collection

router = APIRouter(tags=["collect"])


@router.post("/collect/{item}", status_code=202)
def collect(item: str, request: Request, season_id: str | None = None):
    try:
        job = start_collection(request.app.state, item, season_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return ok({"job_id": job.id})


class CollectAllIn(BaseModel):
    names: list[str] | None = Field(default=None, max_length=5000)   # only these tracked items; all when omitted


@router.post("/collect-all", status_code=202)
def collect_all(request: Request, body: CollectAllIn | None = None):
    from ..services.cost import collect_estimate
    from ..services.initializer import run_collect_all
    st = request.app.state
    items = st.items.list(enabled_only=True)
    names = (body or CollectAllIn()).names
    if names is not None:
        wanted = set(names)
        items = [it for it in items if it["name"] in wanted]
    if not items:
        raise HTTPException(422, "没有可采集的道具 / no tracked items to collect")
    if st.jobs.find_active("collect"):
        raise HTTPException(409, "已有采集正在进行 / a collection is already running")
    label = "采集全部道具 / collect all items" if names is None else f"采集 {len(items)} 个道具 / collect {len(items)} items"
    job = st.jobs.create("collect", label, meta={"item": "*"})
    st.jobs.set_estimate(job, sum(collect_estimate(st, it)["requests"] for it in items))
    chosen = [it["name"] for it in items]
    st.jobs.run_in_thread(job, lambda j: run_collect_all(st, j, chosen))
    return ok({"job_id": job.id})
