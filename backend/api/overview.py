from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services.overview import build_overview

router = APIRouter(tags=["overview"])


@router.get("/overview")
def overview(request: Request, season_id: str | None = None, all: bool = False):
    """Tracked items grouped by type with latest price, 24h change and a week's sparkline per quality (one call),
    for one season: the chosen one, or the current one when none is given."""
    try:
        return ok(build_overview(request.app.state, season_id, all_seasons=all))
    except KeyError:
        raise HTTPException(404, f"season {season_id} not found")
