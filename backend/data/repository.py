from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from ..settings import Settings

_ITEM_RE = re.compile(r"^[\w][\w '’\-():,.&]{0,79}$")


def validate_item(name: str) -> str:
    if not _ITEM_RE.match(name or ""):
        raise ValueError(f"invalid item name {name!r}")
    return name


def table_name(item: str) -> str:
    """Postgres table / CSV stem for an item: lowercase slug, never starting with a digit."""
    s = re.sub(r"[^a-z0-9]+", "_", validate_item(item).lower()).strip("_")
    return s if s and not s[0].isdigit() else f"item_{s}"


@dataclass
class ItemInfo:
    name: str
    rows: int
    first_ts: str | None
    last_ts: str | None


@dataclass(frozen=True)
class ListingQuery:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    sold: bool | None = None
    min_qty: int | None = None
    max_qty: int | None = None
    page: int = 1
    page_size: int = 100


class ListingRepository(Protocol):
    def list_items(self) -> list[ItemInfo]: ...
    def fetch(self, item: str, q: ListingQuery) -> tuple[list[dict], int]: ...
    def fetch_all(self, item: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame: ...
    def source_name(self) -> str: ...


def make_repository(settings: Settings, log: Callable[[str], None] = print) -> ListingRepository:
    from .csv_repo import CsvListingRepository
    if settings.postgres_configured:
        from .postgres_repo import PostgresListingRepository
        try:
            repo = PostgresListingRepository(settings)
            repo.probe()
            log(f"data source: postgres {settings.db_host}:{settings.db_port}/{settings.db_database}")
            return repo
        except Exception as exc:
            log(f"postgres unavailable ({exc}); falling back to CSV")
    else:
        log("postgres not configured (DB_USER empty); using CSV")
    return CsvListingRepository(settings.csv_dirs, settings.items_config_path)
