from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .repository import ItemInfo, ListingQuery, validate_item

COLUMNS = ["item_id", "item", "quantity", "price_per_unit", "price", "has_sold", "created_at"]


class CsvListingRepository:
    def __init__(self, csv_dir: Path, items_config_path: Path):
        self._dir = Path(csv_dir)
        self._config_path = Path(items_config_path)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def source_name(self) -> str:
        return "csv"

    def _files(self) -> dict[str, Path]:
        if not self._config_path.exists():
            return {}
        cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
        return {it["name"]: self._dir / it["csv"] for it in cfg.get("items", []) if it.get("enabled", True)}

    def _frame(self, item: str) -> pd.DataFrame:
        files = self._files()
        if item not in files or not files[item].exists():
            raise KeyError(f"no CSV data for item {item!r}")
        path = files[item]
        mtime = path.stat().st_mtime
        cached = self._cache.get(item)
        if cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["has_sold"] = df["has_sold"].astype(str).str.lower().isin(["true", "1", "t"])
        df = df.dropna(subset=["created_at"]).sort_values("created_at", ascending=False).reset_index(drop=True)
        self._cache[item] = (mtime, df)
        return df

    def list_items(self) -> list[ItemInfo]:
        out = []
        for name in self._files():
            try:
                df = self._frame(name)
            except KeyError:
                out.append(ItemInfo(name, 0, None, None))
                continue
            out.append(ItemInfo(name, len(df),
                                df["created_at"].min().isoformat() if len(df) else None,
                                df["created_at"].max().isoformat() if len(df) else None))
        return out

    def _filtered(self, item: str, start, end, sold=None, min_qty=None, max_qty=None) -> pd.DataFrame:
        df = self._frame(validate_item(item))
        m = pd.Series(True, index=df.index)
        if start is not None:
            m &= df["created_at"] >= start
        if end is not None:
            m &= df["created_at"] < end
        if sold is not None:
            m &= df["has_sold"] == sold
        if min_qty is not None:
            m &= df["quantity"] >= min_qty
        if max_qty is not None:
            m &= df["quantity"] <= max_qty
        return df[m]

    def fetch(self, item: str, q: ListingQuery) -> tuple[list[dict], int]:
        df = self._filtered(item, q.start, q.end, q.sold, q.min_qty, q.max_qty)
        total = len(df)
        page = df.iloc[(q.page - 1) * q.page_size: q.page * q.page_size].copy()
        page["created_at"] = page["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        page["item_id"] = page["item_id"].astype(str)
        return page.to_dict(orient="records"), total

    def fetch_all(self, item: str, start, end) -> pd.DataFrame:
        return self._filtered(item, start, end).sort_values("created_at")
