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
    from src.api.storage import item_slug
    return item_slug(validate_item(item))


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
    from .sqlite_repo import SqliteListingRepository, import_csv_once
    if settings.postgres_configured:
        from .postgres_repo import PostgresListingRepository
        try:
            repo = PostgresListingRepository(settings)
            repo.probe()
            log(f"数据库：Postgres {settings.db_host}:{settings.db_port}/{settings.db_database}")
            return repo
        except Exception as exc:
            log(f"连接不上 Postgres（{exc}），改用内置数据库")
    imported = import_csv_once(settings)
    for name, n in imported.items():
        log(f"已把 {name} 的 {n} 条旧 CSV 数据导入内置数据库")
    log(f"数据库：内置数据库 {settings.sqlite_path}")
    return SqliteListingRepository(settings.sqlite_path, settings.items_config_path)
