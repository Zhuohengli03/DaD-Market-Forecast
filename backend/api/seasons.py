from fastapi import APIRouter, HTTPException, Request

from ..data.seasons import SeasonError
from ..schemas import SeasonIn, ok

router = APIRouter(tags=["seasons"])


@router.get("/seasons")
def list_seasons(request: Request):
    return ok(request.app.state.seasons.list())


@router.post("/seasons", status_code=201)
def add_season(body: SeasonIn, request: Request):
    try:
        return ok(request.app.state.seasons.add(body))
    except SeasonError as exc:
        raise HTTPException(422, str(exc))


@router.put("/seasons/{season_id}")
def update_season(season_id: str, body: SeasonIn, request: Request):
    try:
        return ok(request.app.state.seasons.update(season_id, body))
    except SeasonError as exc:
        raise HTTPException(422, str(exc))
    except KeyError:
        raise HTTPException(404, f"season {season_id} not found")


@router.delete("/seasons/{season_id}")
def delete_season(season_id: str, request: Request):
    try:
        request.app.state.seasons.delete(season_id)
    except KeyError:
        raise HTTPException(404, f"season {season_id} not found")
    return ok({"deleted": season_id})
