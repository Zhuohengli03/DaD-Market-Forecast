"""Where collected listings go: Postgres when configured and reachable, otherwise a CSV file in data/."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

CSV_COLUMNS = ["item_id", "item", "quantity", "price_per_unit", "price", "has_sold", "created_at", "insert_time"]


def item_slug(name: str) -> str:
    """File/table stem for an item name: lowercase, non-alphanumerics to '_', never starting with a digit."""
    import re
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s if s and not s[0].isdigit() else f"item_{s}"


def record_key(item_id, created_at) -> str:
    return f"{item_id}_{created_at}"


class _FileLock:
    """Cross-process lock via an exclusively created file; portable (no fcntl)."""

    def __init__(self, path: Path, timeout: float = 120.0):
        self.path, self.timeout = Path(path), timeout

    def __enter__(self):
        import time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.path} is held by another collector; delete it if no collector is running")
                time.sleep(0.05)

    def __exit__(self, *exc):
        os.close(self._fd)
        os.unlink(self.path)


class CsvStorage:
    name = "csv"

    def __init__(self, path: Path):
        self.path = Path(path)

    def existing_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as fh:
            return {record_key(r["item_id"], r["created_at"]) for r in csv.DictReader(fh)}

    def latest_created_at(self) -> str | None:
        if not self.path.exists():
            return None
        with self.path.open(newline="", encoding="utf-8") as fh:
            stamps = [r["created_at"] for r in csv.DictReader(fh) if r.get("created_at")]
        return max(stamps) if stamps else None

    def insert(self, rows: list[dict]) -> int:
        with _FileLock(self.path.with_suffix(self.path.suffix + ".lock")):
            return self._insert_locked(rows)

    def _insert_locked(self, rows: list[dict]) -> int:
        known = self.existing_keys()
        fresh = [r for r in rows if record_key(r["id"], r["created_at"]) not in known]
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        now = datetime.now().isoformat(sep=" ")
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if new_file:
                w.writeheader()
            for r in fresh:
                w.writerow({"item_id": r["id"], "item": r["item"], "quantity": r["quantity"],
                            "price_per_unit": r["price_per_unit"], "price": r["price"], "has_sold": r["has_sold"],
                            "created_at": r["created_at"], "insert_time": now})
        return len(fresh)

    def close(self) -> None:
        pass


class PostgresStorage:
    name = "postgres"

    def __init__(self, table: str):
        from src.database.Database_connect import DarkerMarketDB
        self.db = DarkerMarketDB(items=table)
        if not self.db.connect():
            raise ConnectionError("database connection failed")

    def existing_keys(self) -> set[str]:
        cur = self.db.connector.cursor()
        cur.execute(f"SELECT item_id, created_at FROM {self.db.items}")
        keys = {record_key(a, b) for a, b in cur.fetchall()}
        cur.close()
        return keys

    def latest_created_at(self) -> str | None:
        cur = self.db.connector.cursor()
        cur.execute(f"SELECT MAX(created_at::timestamptz) FROM {self.db.items}")
        value = cur.fetchone()[0]
        cur.close()
        return value.strftime("%Y-%m-%dT%H:%M:%SZ") if value else None

    def insert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        self.db.add_data(rows)
        return int(self.db.batch_insert_all_data() or 0)

    def close(self) -> None:
        self.db.disconnect()


def open_storage(table: str, data_dir: Path, log=print):
    """Postgres if DB_USER is set and the server answers; otherwise data/<table>.csv."""
    if os.getenv("DB_USER"):
        try:
            s = PostgresStorage(table)
            log(f"数据存储：Postgres 表 {table}")
            return s
        except Exception as exc:
            log(f"连接不上 Postgres（{exc}），改为写入 CSV")
    path = Path(data_dir) / f"{table}.csv"
    log(f"数据存储：{path}")
    return CsvStorage(path)
