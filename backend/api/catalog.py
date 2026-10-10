from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient
from ..schemas import ok
from ..services.catalog_sync import check_tradability, sync_catalog

router = APIRouter(tags=["catalog"])


def _client(request: Request):
    st = request.app.state
    key = st.settings.darker_market_api_key
    if not key:
        raise HTTPException(409, "还没有设置 DarkerDB API key，请先在设置页填写 / Set your DarkerDB API key first")
    factory = getattr(st, "client_factory", None) or (lambda k: DarkerDBClient(k))
    return factory(key)


def _progress(jobs, job):
    return lambda stage, done=None, total=None, **d: jobs.progress(job, stage, done, total, detail=d)


@router.get("/catalog")
def catalog(request: Request, q: str = "", item_type: str = "", has_stats: bool | None = None,
            page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    store = request.app.state.catalog
    rows, total, facets = store.query(q, item_type, has_stats, page, page_size)
    return ok(rows, meta={"total": total, "page": page, "page_size": page_size, "facets": facets, **store.meta()})


@router.get("/catalog/meta")
def catalog_meta(request: Request):
    return ok(request.app.state.catalog.meta())


class CheckIn(BaseModel):
    force: bool = False


@router.post("/catalog/sync", status_code=202)
def start_sync(request: Request):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库任务正在运行 / a catalog job is already running")
    job = st.jobs.create("catalog", "同步道具库 / sync item catalog")
    st.jobs.run_in_thread(job, lambda j: str(sync_catalog(client, st.catalog, lambda s: st.jobs.append_log(j, s),
                                                          progress=_progress(st.jobs, j))))
    return ok({"job_id": job.id})


@router.post("/catalog/check-trade", status_code=202)
def start_trade_check(request: Request, body: CheckIn | None = None):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库任务正在运行 / a catalog job is already running")
    if not st.catalog.count():
        raise HTTPException(409, "请先同步道具库 / sync the catalog first")
    job = st.jobs.create("catalog", "检查哪些道具能交易 / check which items trade")
    force = (body or CheckIn()).force
    st.jobs.run_in_thread(job, lambda j: str(check_tradability(client, st.catalog, lambda s: st.jobs.append_log(j, s), force,
                                                               progress=_progress(st.jobs, j))))
    return ok({"job_id": job.id})
