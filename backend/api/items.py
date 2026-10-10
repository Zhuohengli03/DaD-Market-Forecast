from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..schemas import ok

router = APIRouter(tags=["items"])


class TrackIn(BaseModel):
    item_id: str = Field(pattern=r"^id\.item\.[A-Za-z0-9_]+$")
    category: str = ""


@router.get("/items")
def list_items(request: Request):
    st = request.app.state
    stats = {i.name: i for i in st.item_stats.get()}
    out = []
    for it in st.items.list():
        s = stats.get(it["name"])
        out.append({**it, "rows": s.rows if s else 0, "first_ts": s.first_ts if s else None,
                    "last_ts": s.last_ts if s else None})
    return ok(out)


@router.post("/items/track", status_code=201)
def track(body: TrackIn, request: Request):
    st = request.app.state
    raw = next((r for r in st.catalog.read()["items"] if r.get("id") == body.item_id), None)
    if raw is None:
        raise HTTPException(404, f"道具库里没有 {body.item_id}，请先同步道具库 / not in the synced catalog")
    name = str(raw.get("name") or body.item_id)
    try:
        entry = st.items.track(name, body.item_id, body.category or str(raw.get("item_type", "")),
                               str(raw.get("rarity", "")))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    st.item_stats.invalidate()
    return ok(entry)


@router.delete("/items/{name}")
def untrack(name: str, request: Request):
    st = request.app.state
    try:
        st.items.untrack(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    st.item_stats.invalidate()
    return ok({"untracked": name})
