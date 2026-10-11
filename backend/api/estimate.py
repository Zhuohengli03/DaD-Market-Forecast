"""Request-cost estimates shown before the user starts anything that calls DarkerDB."""
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..schemas import ok
from ..services import cost

router = APIRouter(tags=["estimate"])


@router.get("/estimate/collect-all")
def estimate_collect_all(request: Request, season_id: str | None = None):
    try:
        return ok(cost.collect_all_estimate(request.app.state, season_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'"))


@router.get("/estimate/collect/{name}")
def estimate_collect(name: str, request: Request, season_id: str | None = None):
    from ..services.collection import collect_range, market_not_open, season_progress
    st = request.app.state
    try:
        entry = st.items.get(name)
        since, until = collect_range(st, name, season_id)
        opens = market_not_open(st, season_id)
        progress = season_progress(st, name, season_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'"))
    if opens or progress["complete"]:   # the market has not opened yet, or the season is already in full
        return ok({"requests": 0, "known": True, "since": since, "until": until, "opens": opens, "progress": progress})
    return ok({**cost.collect_estimate(st, entry, since=since, until=until), "since": since, "until": until, "opens": None,
               "progress": progress})


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
