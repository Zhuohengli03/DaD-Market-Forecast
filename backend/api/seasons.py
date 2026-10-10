from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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


class ImportIn(BaseModel):
    replace: bool = False


def darkerdb_seasons(request: Request) -> list[dict]:
    """Season boundaries from DarkerDB's patch list: each season ends where the next one starts."""
    from .catalog import _client
    client = _client(request)
    rows = []
    for page in client.iter_pages("/patches", {"kind": "season", "limit": 250}):
        rows.extend(r for r in page if r.get("kind") == "season" and r.get("released_at"))
    rows.sort(key=lambda r: r["released_at"])
    out = []
    for i, r in enumerate(rows):
        number = r.get("number") or r["slug"].rsplit("-", 1)[-1]
        start = f"{r['released_at'][:10]}T00:00:00+00:00"
        end = f"{rows[i + 1]['released_at'][:10]}T00:00:00+00:00" if i + 1 < len(rows) else None
        out.append({"id": r["slug"], "name": f"Season {number}", "start": start, "end": end,
                    "note": r.get("title", ""), "is_current": bool(r.get("is_current"))})
    return out


@router.get("/seasons/darkerdb")
def preview_darkerdb_seasons(request: Request):
    return ok(darkerdb_seasons(request))


@router.post("/seasons/import")
def import_darkerdb_seasons(body: ImportIn, request: Request):
    store = request.app.state.seasons
    fetched = darkerdb_seasons(request)
    added, skipped = store.import_records(fetched, replace=body.replace)
    return ok({"added": added, "skipped": skipped, "seasons": store.list()})
