from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient
from ..schemas import ok
from ..services.catalog_sync import scan_market, sync_catalog

router = APIRouter(tags=["catalog"])
DEFAULT_SCAN_PAGES = 40


class ScanIn(BaseModel):
    max_pages: int = Field(default=DEFAULT_SCAN_PAGES, ge=1, le=600)


def _client(request: Request):
    st = request.app.state
    key = st.settings.darker_market_api_key
    if not key:
        raise HTTPException(409, "还没有设置 DarkerDB API key，请先在设置页填写 / Set your DarkerDB API key first")
    factory = getattr(st, "client_factory", None) or (lambda k: DarkerDBClient(k))
    return factory(key)


def scan_since(seasons) -> str:
    """Start of the latest season if one is configured, else the last 7 days."""
    latest = seasons.list()[-1:] if seasons.list() else []
    if latest:
        return latest[0]["start"]
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")


@router.get("/catalog")
def catalog(request: Request, q: str = "", rarity: str = "", item_type: str = "", tradable: bool | None = None,
            page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    store = request.app.state.catalog
    rows, total, facets = store.query(q, rarity, item_type, tradable, page, page_size)
    return ok(rows, meta={"total": total, "page": page, "page_size": page_size, "facets": facets, **store.meta()})


@router.get("/catalog/meta")
def catalog_meta(request: Request):
    return ok(request.app.state.catalog.meta())


@router.post("/catalog/sync", status_code=202)
def start_sync(request: Request):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库正在同步中 / catalog sync already running")
    job = st.jobs.create("catalog", "同步道具库 / sync item catalog")
    st.jobs.run_in_thread(job, lambda j: str(sync_catalog(client, st.catalog, lambda s: st.jobs.append_log(j, s))))
    return ok({"job_id": job.id})


@router.post("/catalog/scan-market", status_code=202)
def start_scan(request: Request, body: ScanIn | None = None):
    st = request.app.state
    client = _client(request)
    if st.jobs.find_active("catalog"):
        raise HTTPException(409, "道具库任务正在运行 / a catalog job is already running")
    since = scan_since(st.seasons)
    pages = (body or ScanIn()).max_pages
    job = st.jobs.create("catalog", f"扫描市场 / scan market since {since[:10]}")
    st.jobs.run_in_thread(job, lambda j: str(scan_market(client, st.catalog, since, pages,
                                                         lambda s: st.jobs.append_log(j, s))))
    return ok({"job_id": job.id})
