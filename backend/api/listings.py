from fastapi import APIRouter, HTTPException, Query, Request

from ..data.repository import ListingQuery
from ..schemas import ok
from .series import item_param, resolve_range

router = APIRouter(tags=["listings"])


@router.get("/listings/{item}")
def listings(item: str, request: Request, season_id: str | None = None, start: str | None = None,
             end: str | None = None, sold: bool | None = None,
             min_qty: int | None = Query(None, ge=0), max_qty: int | None = Query(None, ge=0),
             page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500)):
    item = item_param(item)
    s, e = resolve_range(request, season_id, start, end)
    q = ListingQuery(start=s, end=e, sold=sold, min_qty=min_qty, max_qty=max_qty, page=page, page_size=page_size)
    try:
        rows, total = request.app.state.repo.fetch(item, q)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return ok(rows, meta={"total": total, "page": page, "page_size": page_size})
