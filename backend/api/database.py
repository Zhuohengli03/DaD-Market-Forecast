"""The local database page: browse every stored listing and delete some. There is deliberately no way to edit."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..schemas import ok

router = APIRouter(tags=["database"])


def _ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except ValueError:
        raise HTTPException(422, f"时间格式不对 / bad time {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _repo(request: Request):
    repo = request.app.state.repo
    if not hasattr(repo, "stored_items"):
        raise HTTPException(501, "当前数据源不支持在界面里浏览 / this data source cannot be browsed here")
    return repo


def _chinese_names(st) -> dict[str, str]:
    """English item name -> Chinese, from the catalog: tracked items by their exact variant, others by name."""
    from ..data.catalog import variant_name_zh
    out: dict[str, str] = {}
    for variants in st.catalog.types().values():
        for v in variants:
            zh = variant_name_zh(v, variants)
            if zh and v.get("name"):
                out.setdefault(v["name"], zh)
    for it in st.items.list():
        found = st.catalog.variant(it["item_id"]) if it.get("item_id") else None
        zh = variant_name_zh(*found) if found else None
        if zh:
            out[it["name"]] = zh
    return out


@router.get("/db/items")
def stored_items(request: Request):
    repo = _repo(request)
    zh = _chinese_names(request.app.state)
    rows = [{**i, "name_zh": zh.get(i["name"]) or zh.get(i["name"].split(" (")[0])} for i in repo.stored_items()]
    return ok(rows, meta={"source": repo.source_name()})


@router.get("/db/rows")
def rows(request: Request, key: str = Query(min_length=1), start: str | None = None, end: str | None = None,
         page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500)):
    data, total = _repo(request).browse(key, _ts(start), _ts(end), page, page_size)
    return ok(data, meta={"total": total, "page": page, "page_size": page_size})


class DeleteIn(BaseModel):
    key: str = Field(min_length=1)
    ids: list[str] = Field(default_factory=list, max_length=500)
    start: str | None = None
    end: str | None = None
    all: bool = False          # the whole item: must be said explicitly


@router.post("/db/delete")
def delete(body: DeleteIn, request: Request):
    st = request.app.state
    repo = _repo(request)
    if not (body.ids or body.start or body.end or body.all):
        raise HTTPException(422, "没有选要删除的内容 / choose rows, a time range or all")
    item = next((i for i in repo.stored_items() if i["key"] == body.key), None)
    if item and st.jobs.find_active("collect", item=item["name"]):
        raise HTTPException(409, f"{item['name']} 正在采集，等采集结束再删除 / collection running")
    n = repo.delete(body.key, _ts(body.start), _ts(body.end), body.ids or None)
    if n:
        st.item_stats.invalidate()
        # season coverage marks no longer hold: the next collection walks the season again and fills the hole
        (st.settings.data_dir / "coverage" / f"{body.key}.json").unlink(missing_ok=True)
    return ok({"deleted": n})
