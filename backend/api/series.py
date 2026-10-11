from fastapi import APIRouter, HTTPException, Query, Request

from ..data.repository import validate_item
from ..schemas import ok
from ..services.series import RESOLUTIONS, get_series, parse_ts, season_range

router = APIRouter(tags=["series"])


def item_param(item: str) -> str:
    try:
        return validate_item(item)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def resolve_range(request: Request, season_id: str | None, start: str | None, end: str | None):
    try:
        s, e = season_range(request.app.state.seasons, season_id) if season_id else (None, None)
    except KeyError:
        raise HTTPException(404, f"season {season_id} not found")
    try:
        requested_start, requested_end = parse_ts(start), parse_ts(end)
        # A custom range may narrow a selected season, never widen it across SSF/wipes.
        s = max(x for x in (s, requested_start) if x is not None) if s is not None or requested_start is not None else None
        e = min(x for x in (e, requested_end) if x is not None) if e is not None or requested_end is not None else None
    except Exception as exc:
        raise HTTPException(422, f"bad timestamp: {exc}")
    if s is not None and e is not None and s >= e:
        raise HTTPException(422, "start must be before end within the season's market range")
    return s, e


@router.get("/series/{item}/seasons")
def series_by_season(item: str, request: Request):
    from ..services.series import season_compare
    st = request.app.state
    return ok({"seasons": season_compare(st.repo, item_param(item), st.seasons.list())})


@router.get("/series/{item}")
def series(item: str, request: Request, season_id: str | None = None, start: str | None = None,
           end: str | None = None, resolution: str = Query("1h")):
    item = item_param(item)
    if resolution not in RESOLUTIONS:
        raise HTTPException(422, f"resolution must be one of {RESOLUTIONS}")
    s, e = resolve_range(request, season_id, start, end)
    try:
        return ok(get_series(request.app.state.repo, item, s, e, resolution))
    except KeyError as exc:
        raise HTTPException(404, str(exc))
