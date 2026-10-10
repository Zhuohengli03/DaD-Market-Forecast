from fastapi import APIRouter, Request

from ..schemas import ok
from ..services.overview import build_overview

router = APIRouter(tags=["overview"])


@router.get("/overview")
def overview(request: Request):
    """Tracked items grouped by type with latest price, 24h change and a 48h sparkline per quality (one call)."""
    return ok(build_overview(request.app.state))
