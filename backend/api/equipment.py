from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from ..schemas import ok
from ..services.equipment import build_market_params, query_market, stats_for_type
from .catalog import _client

router = APIRouter(tags=["equipment"])


class StatFilter(BaseModel):
    kind: Literal["primary", "secondary"]
    name: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.name}: min must not exceed max")
        return self


class EquipmentQuery(BaseModel):
    archetype: str = Field(pattern=r"^id\.item\.[a-z0-9_]+$")
    rarity: str | None = Field(default=None, pattern=r"^[a-z]+$")
    filters: list[StatFilter] = Field(default_factory=list, max_length=12)
    sold_only: bool = False
    max_pages: int = Field(default=10, ge=1, le=40)
    since: str | None = None


def _equipment_type(request: Request, archetype: str) -> list[dict]:
    from ..data.catalog import has_stats
    variants = request.app.state.catalog.type_variants(archetype)
    if not variants:
        raise HTTPException(404, "道具库里没有这个类型，请先同步道具库 / type not in the synced catalog")
    if not has_stats(variants):
        raise HTTPException(422, "这个道具没有随机词条，请在价格走势和预测里查看 / this item has no random stats; use price history")
    return variants


@router.get("/equipment/{archetype}/stats")
def equipment_stats(archetype: str, request: Request):
    variants = _equipment_type(request, archetype)
    stats = stats_for_type(_client(request), archetype)
    return ok({"stats": stats, "rarities": [v.get("rarity") for v in variants]})


@router.post("/equipment/query")
def equipment_query(body: EquipmentQuery, request: Request):
    _equipment_type(request, body.archetype)
    seasons = request.app.state.seasons.list()
    since = body.since or (seasons[-1]["start"] if seasons else None)
    params = build_market_params(body.archetype, body.rarity, [f.model_dump() for f in body.filters], body.sold_only, since)
    result = query_market(_client(request), params, body.max_pages)
    return ok({**result, "since": since, "params": params})
