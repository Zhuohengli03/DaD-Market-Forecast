from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

ITEM_NAME_RE = r"^[A-Za-z0-9 ]{1,64}$"


def ok(data: Any, meta: dict | None = None) -> dict:
    return {"ok": True, "data": data, "error": None, "meta": meta}


def fail(error: str, meta: dict | None = None) -> dict:
    return {"ok": False, "data": None, "error": error, "meta": meta}


class SeasonIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    start: str = Field(min_length=1)
    end: str | None = None
    note: str = ""


class ForecastIn(BaseModel):
    item: str = Field(pattern=ITEM_NAME_RE)
    season_id: str = Field(min_length=1)
    horizon_hours: int = Field(default=72, ge=24, le=168)
    models: list[str] = Field(default_factory=list)
    coverage: float = Field(default=0.9, ge=0.5, le=0.99)


class RunTaskIn(BaseModel):
    script_path: str = Field(min_length=1)


class ItemConfigEntry(BaseModel):
    name: str = Field(pattern=ITEM_NAME_RE)
    item_id: str | None = Field(default=None, pattern=r"^id\.item\.[A-Za-z0-9_]+$")
    file: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_]+\.py$")
    csv: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_]+\.csv$")
    category: str = "ore"
    description: str = ""
    enabled: bool = True


class ItemsConfigIn(BaseModel):
    items: list[ItemConfigEntry]
    categories: dict[str, str] = Field(default_factory=dict)
    auto_discovery: dict[str, Any] = Field(default_factory=dict)

    @field_validator("items")
    @classmethod
    def unique_names(cls, v: list[ItemConfigEntry]) -> list[ItemConfigEntry]:
        names = [i.name for i in v]
        if len(names) != len(set(names)):
            raise ValueError("duplicate item names")
        return v
