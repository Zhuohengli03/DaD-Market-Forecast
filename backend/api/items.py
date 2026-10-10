import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from ..data.app_settings import MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS

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


_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ScheduleIn(BaseModel):
    mode: Literal["default", "interval", "daily"]
    seconds: int | None = None
    times: list[str] | None = None

    @model_validator(mode="after")
    def check(self):
        if self.mode == "interval" and not (self.seconds and MIN_INTERVAL_SECONDS <= self.seconds <= MAX_INTERVAL_SECONDS):
            raise ValueError("采集间隔需要在 30 秒到 30 天之间 / interval must be between 30 seconds and 30 days")
        if self.mode == "daily":
            if not self.times or len(self.times) > 24 or not all(_TIME.match(t) for t in self.times):
                raise ValueError("请填写 1~24 个 HH:MM 格式的时间 / give 1-24 times as HH:MM")
        return self


@router.put("/items/{name}/schedule")
def set_item_schedule(name: str, body: ScheduleIn, request: Request):
    st = request.app.state
    sched = None if body.mode == "default" else (
        {"mode": "interval", "seconds": body.seconds} if body.mode == "interval"
        else {"mode": "daily", "times": sorted(set(body.times or []))})
    try:
        entry = st.items.set_schedule(name, sched)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    st.scheduler.restart_if_running()
    return ok(entry)


class BulkTrackIn(BaseModel):
    item_ids: list[str] = Field(max_length=3000)


@router.post("/items/track-bulk")
def track_bulk(body: BulkTrackIn, request: Request):
    from ..data.catalog import has_stats
    st = request.app.state
    added, skipped = 0, {"has_stats": 0, "already_tracked": 0, "not_in_catalog": 0}
    for iid in body.item_ids:  # a repeat counts as already tracked
        found = st.catalog.variant(iid)
        if found is None:
            skipped["not_in_catalog"] += 1
            continue
        raw, variants = found
        if has_stats(variants):
            skipped["has_stats"] += 1
            continue
        name = str(raw.get("name") or iid) + (f" ({raw.get('rarity')})" if len(variants) > 1 else "")
        try:
            st.items.track(name, iid, str(raw.get("item_type", "")))
            added += 1
        except ValueError:
            skipped["already_tracked"] += 1
    st.item_stats.invalidate()
    st.scheduler.restart_if_running()
    return ok({"added": added, "skipped": skipped})
