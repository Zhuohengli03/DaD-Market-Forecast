from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient
from src.season_calendar import market_start
from ..schemas import ok
from ..services.catalog_sync import check_tradability, sync_catalog, type_history

router = APIRouter(tags=["catalog"])


def client_for_state(st):
    key = st.settings.darker_market_api_key
    if not key:
        raise HTTPException(409, "还没有设置 DarkerDB API key，请先在设置页填写 / Set your DarkerDB API key first")
    factory = getattr(st, "client_factory", None) or (lambda k: DarkerDBClient(k, quota_file=st.settings.data_dir / "quota.json"))
    return factory(key)


def _client(request: Request):
    st = request.app.state
    key = st.settings.darker_market_api_key
    if not key:
        raise HTTPException(409, "还没有设置 DarkerDB API key，请先在设置页填写 / Set your DarkerDB API key first")
    factory = getattr(st, "client_factory", None) or (lambda k: DarkerDBClient(k, quota_file=st.settings.data_dir / "quota.json"))
    return factory(key)


def _counted(st, client, fn):
    """Run a job body and record the DarkerDB credits its client spent, even if it fails."""
    def run(job):
        try:
            return fn(job)
        finally:
            st.jobs.add_credits(job, getattr(client, "credits_used", 0))
    return run


def _progress(jobs, job):
    return lambda stage, done=None, total=None, **d: jobs.progress(job, stage, done, total, detail=d)


@router.get("/catalog")
def catalog(request: Request, q: str = "", item_type: str = "", has_stats: bool | None = None,
            page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=1000)):
    st = request.app.state
    rows, total, facets = st.catalog.query(q, item_type, has_stats, page, page_size, season_start=latest_season_start(st))
    return ok(rows, meta={"total": total, "page": page, "page_size": page_size, "facets": facets, **_meta(st)})


def _meta(st) -> dict:
    """Catalog facts plus the current season: its start and, before the market opens, the opening date."""
    from ..services.collection import market_not_open
    return {**st.catalog.meta(latest_season_start(st)), "market_opens": market_not_open(st)}


@router.get("/catalog/meta")
def catalog_meta(request: Request):
    from ..services.cost import check_cost, sync_cost
    st = request.app.state
    return ok({**_meta(st), "sync_cost": sync_cost(st.catalog), "check_cost": check_cost(st.catalog)})


class CheckIn(BaseModel):
    force: bool = False


@router.post("/catalog/sync", status_code=202)
def start_sync(request: Request):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库任务正在运行 / a catalog job is already running")
    from ..services.cost import sync_cost
    job = st.jobs.create("catalog", "同步道具库 / sync item catalog")
    st.jobs.set_estimate(job, sync_cost(st.catalog))
    st.jobs.run_in_thread(job, _counted(st, client, lambda j: str(sync_catalog(client, st.catalog, lambda s: st.jobs.append_log(j, s),
                                                                               progress=_progress(st.jobs, j)))))
    return ok({"job_id": job.id})


@router.post("/catalog/check-trade", status_code=202)
def start_trade_check(request: Request, body: CheckIn | None = None):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库任务正在运行 / a catalog job is already running")
    if not st.catalog.count():
        raise HTTPException(409, "请先同步道具库 / sync the catalog first")
    from ..services.cost import check_cost
    job = st.jobs.create("catalog", "检查哪些道具能交易 / check which items trade")
    force = (body or CheckIn()).force
    st.jobs.set_estimate(job, len(st.catalog.types()) if force else check_cost(st.catalog))
    st.jobs.run_in_thread(job, _counted(st, client, lambda j: str(check_tradability(client, st.catalog, lambda s: st.jobs.append_log(j, s), force,
                                                                                    progress=_progress(st.jobs, j)))))
    return ok({"job_id": job.id})


def latest_season_start(st) -> str | None:
    import pandas as pd
    starts = [market_start(s) for s in st.seasons.list() if s.get("start")]
    return max(starts, key=lambda x: pd.Timestamp(x)) if starts else None


@router.post("/catalog/types/{archetype}/history")
def one_type_history(archetype: str, request: Request, refresh: bool = False):
    """This season's count and first listing for one type (1 DarkerDB request, then cached; `refresh` asks again)."""
    st = request.app.state
    client = _client(request)
    try:
        info = type_history(client, st.catalog, archetype, latest_season_start(st), refresh=refresh)
        return ok({**info, "credits_used": getattr(client, "credits_used", 0)})
    except KeyError:
        raise HTTPException(404, "道具库里没有这个道具 / unknown item type")


@router.get("/catalog/types/{archetype}")
def one_type(archetype: str, request: Request):
    st = request.app.state
    row = st.catalog.type_row(archetype, latest_season_start(st))
    if row is None:
        raise HTTPException(404, "道具库里没有这个道具 / unknown item type")
    return ok(row)
