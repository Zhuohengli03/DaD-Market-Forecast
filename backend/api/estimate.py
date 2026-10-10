"""Request-cost estimates shown before the user starts anything that calls DarkerDB."""
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from ..schemas import ok
from ..services import cost

router = APIRouter(tags=["estimate"])


@router.get("/estimate/collect-all")
def estimate_collect_all(request: Request):
    return ok(cost.collect_all_estimate(request.app.state))


@router.get("/estimate/collect/{name}")
def estimate_collect(name: str, request: Request):
    st = request.app.state
    return ok(cost.collect_estimate(st, st.items.get(name)))


@router.get("/estimate/schedule")
def estimate_schedule(request: Request, interval_seconds: int | None = Query(None, ge=30, le=30 * 86400)):
    st = request.app.state
    default = interval_seconds or st.app_settings.get()["collect_interval_seconds"]
    return ok(cost.schedule_estimate(st, default))


class ItemsIn(BaseModel):
    item_ids: list[str] = Field(default_factory=list, max_length=3000)
    max_pages: int = Field(cost.MAX_PAGES, ge=1, le=1000)


@router.post("/estimate/items")
def estimate_items(body: ItemsIn, request: Request):
    return ok(cost.items_estimate(request.app.state, body.item_ids, body.max_pages))
