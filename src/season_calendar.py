"""Market opening dates are distinct from wipe/patch dates.

Announcements below give calendar dates, not exact opening instants. We use UTC
day boundaries for date-level filtering, and expose that limitation in the UI.
Never infer SSF duration for an unlisted season.
"""
from datetime import datetime, timezone


# Season market openings and exact hand-over times live in the rules file (src/rules_default.json, or a newer
# synced data/rules.json): they change every season. Sources for each date are listed in the maintainer notes.
def _openings() -> dict:
    from src import rules
    return rules.current()["seasons"].get("market_openings", {})


def _boundaries() -> dict:
    from src import rules
    return rules.current()["seasons"].get("boundaries", {})


def _at_midnight_of(value: str | None, exact: str) -> bool:
    """True when `value` is the date-only (UTC midnight) form of the same day as `exact`."""
    return bool(value) and value[:10] == exact[:10] and value[11:19] in ("00:00:00", "")


def with_market_opening(season: dict) -> dict:
    """Enrich old imported records without overwriting an explicit user value."""
    out = dict(season)
    for field, exact in _boundaries().get(out.get("id"), {}).items():
        if _at_midnight_of(out.get(field), exact):      # a time set by hand is kept
            out[field] = exact
    known = _openings().get(out.get("id"))
    if known and out.get("start", "")[:10] == known[0]:
        out.setdefault("market_open", known[1])
    return out


def market_start(season: dict) -> str:
    season = with_market_opening(season)
    start = season["start"]
    if not season.get("market_open"):
        return start
    opening = season["market_open"] + "T00:00:00+00:00"
    def ts(value):
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t
    return max((start, opening), key=ts)
