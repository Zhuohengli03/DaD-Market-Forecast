from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services import announcements as ann

router = APIRouter(tags=["announcements"])


def steam_get(url, params):
    import requests
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _type_name(st, archetype: str) -> str:
    """English name shared by a type's qualities (Ruby for every ruby), else the tracked item name itself."""
    from ..data.catalog import shared_name
    variants = st.catalog.types().get(archetype)
    if variants:
        return shared_name([v.get("name") or archetype for v in variants])
    if any(it["name"] == archetype for it in st.items.list()):
        return archetype
    raise HTTPException(404, f"没有这个道具 / unknown item {archetype}")


@router.get("/announcements")
def list_announcements(request: Request):
    store = request.app.state.announcements
    items = [{k: a[k] for k in ("id", "date", "title", "url")} for a in store.items()]
    return ok(items, meta={"updated_at": store.updated_at(), "error": getattr(request.app.state, "announcements_error", None)})


def _change_around(st, item: str, date: str, days: int = 3) -> float | None:
    """Median sale price in the `days` after an update over the `days` before it, minus one (None without sales)."""
    import pandas as pd
    at = pd.Timestamp(date)
    try:
        df = st.repo.fetch_all(item, at - pd.Timedelta(days=days), at + pd.Timedelta(days=days))
    except KeyError:
        return None
    if df.empty:
        return None
    sold = df[df["has_sold"].astype(bool)]
    ts = pd.to_datetime(sold["created_at"], utc=True)
    before, after = sold["price_per_unit"][ts < at], sold["price_per_unit"][ts >= at]
    if before.empty or after.empty or not before.median():
        return None
    return float(after.median() / before.median() - 1)


@router.get("/announcements/item/{archetype}")
def item_mentions(archetype: str, request: Request, item: str | None = None):
    """Patches that name this item type; with `item` (a tracked quality), the price change around each one."""
    st = request.app.state
    name = _type_name(st, archetype)
    found = ann.mentions(name, st.announcements.items())
    if item:
        found = [{**m, "change3d": _change_around(st, item, m["date"])} for m in found]
    return ok({"name": name, "mentions": found})


@router.post("/announcements/refresh")
def refresh(request: Request):
    st = request.app.state
    st.announcements_error = ann.refresh(st.announcements, steam_get)
    return ok({"error": st.announcements_error, "count": len(st.announcements.items())})
