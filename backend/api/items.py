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
    zh = {}
    for variants in st.catalog.types().values():
        for v in variants:
            if v.get("name_zh"):
                zh[v["id"]] = v["name_zh"]
                zh.setdefault(v.get("name"), v["name_zh"])
    out = []
    for it in st.items.list():
        s = stats.get(it["name"])
        base = zh.get(it.get("item_id")) or zh.get(it["name"].split(" (")[0])
        suffix = it["name"][len(it["name"].split(" (")[0]):] if base and " (" in it["name"] else ""
        out.append({**it, "name_zh": f"{base}{suffix}" if base else None, "rows": s.rows if s else 0, "first_ts": s.first_ts if s else None,
                    "last_ts": s.last_ts if s else None})
    return ok(out)


@router.post("/items/track", status_code=201)
def track(body: TrackIn, request: Request):
    st = request.app.state
    found = st.catalog.variant(body.item_id)
    if found is None:
        raise HTTPException(404, f"道具库里没有 {body.item_id}，请先同步道具库 / not in the synced catalog")
    raw, variants = found
    from ..data.catalog import has_stats
    if has_stats(variants):
        raise HTTPException(422, "带随机词条的装备不做追踪和预测，请在「装备行情」里按词条查看 / "
                                 "items with random stats are not tracked; use the Equipment market page")
    name = str(raw.get("name") or body.item_id)
    if len(variants) > 1:  # several rarities of one type: the rarity is part of the tracked name
        name = f"{name} ({raw.get('rarity')})"
    try:
        entry = st.items.track(name, body.item_id, body.category or str(raw.get("item_type", "")))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    st.item_stats.invalidate()
    st.scheduler.restart_if_running()
    return ok(entry)


@router.delete("/items/{name}")
def untrack(name: str, request: Request):
    st = request.app.state
    try:
        st.items.untrack(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    st.item_stats.invalidate()
    st.scheduler.restart_if_running()
    return ok({"untracked": name})
