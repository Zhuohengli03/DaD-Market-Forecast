"""Built-in database: a single SQLite file (data/market.db). No server, no install, created on demand."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    item           TEXT NOT NULL,      -- tracked item key (slug of its name, e.g. iron_ore)
    item_id        TEXT NOT NULL,      -- DarkerDB listing id
    item_name      TEXT,
    quantity       INTEGER NOT NULL,
    price_per_unit REAL NOT NULL,
    price          REAL NOT NULL,
    has_sold       INTEGER NOT NULL,
    created_at     TEXT NOT NULL,      -- UTC, 2026-01-01T00:00:00Z
    insert_time    TEXT NOT NULL,
    UNIQUE (item, item_id, created_at)
);
CREATE INDEX IF NOT EXISTS idx_listings_item_created ON listings (item, created_at);
CREATE TABLE IF NOT EXISTS imports (
    source      TEXT PRIMARY KEY,      -- legacy CSV already imported
    rows        INTEGER NOT NULL,
    imported_at TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")   # readers do not block the collector and vice versa
    con.executescript(SCHEMA)
    return con
