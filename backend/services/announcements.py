"""Steam patch notes for Dark and Darker, kept locally, searched for item names (no DarkerDB credits)."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

STEAM_NEWS = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
REFRESH_HOURS = 6
MAX_PAGES = 60          # ~6,000 posts; a safety stop if Steam ever ignores enddate


def _words(key: str) -> re.Pattern:
    """Title words from the rules file (announcements.patch / not_patch), as one case-insensitive pattern."""
    from src import rules
    words = rules.current()["announcements"].get(key) or []
    return re.compile("|".join(re.escape(w) for w in words) or r"(?!)", re.I)


def _app_and_oldest() -> tuple[int, int]:
    from src import rules
    a = rules.current()["announcements"]
    oldest = datetime.fromisoformat(a.get("oldest", "2024-09-01T00:00:00Z").replace("Z", "+00:00"))
    return int(a.get("steam_app_id", 2016590)), int(oldest.timestamp())


def is_patch(title: str) -> bool:
    return bool(_words("patch").search(title)) and not _words("not_patch").search(title)


def plain_text(bbcode: str) -> str:
    s = re.sub(r"\[img\].*?\[/img\]", "", bbcode, flags=re.S | re.I)
    s = re.sub(r"\[\*\]", "\n- ", s)
    s = re.sub(r"\[/?(h\d|list|olist|p)\]", "\n", s, flags=re.I)
    s = re.sub(r"\[/?[a-z0-9]+(=[^\]]*)?\]", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    lines = [ln.strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_steam(get, since_ts: int | None) -> list[dict]:
    """Page back from the newest post until `since_ts` (or season 4); keep patch posts only."""
    out, seen, end = [], set(), None
    app_id, oldest = _app_and_oldest()
    stop = since_ts or oldest
    for _ in range(MAX_PAGES):
        params = {"appid": app_id, "count": 100, "maxlength": 0, "format": "json",
                  "feeds": "steam_community_announcements"}
        if end is not None:
            params["enddate"] = end
        posts = (get(STEAM_NEWS, params).get("appnews") or {}).get("newsitems") or []
        if not posts:
            break
        for p in posts:
            if p["date"] < stop:
                return out
            if str(p["gid"]) in seen:
                continue
            seen.add(str(p["gid"]))
            if is_patch(p.get("title", "")):
                out.append({"id": str(p["gid"]), "date": _iso(p["date"]), "title": p["title"],
                            "url": p.get("url", ""), "text": plain_text(p.get("contents", ""))})
        oldest = min(p["date"] for p in posts) - 1
        if end is not None and oldest >= end:      # the same page again: Steam ignored enddate
            break
        end = oldest
    return out


class AnnouncementStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def _read(self) -> dict:
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def items(self) -> list[dict]:
        kept = [a for a in self._read().get("items", []) if is_patch(a.get("title", ""))]   # the filter may have grown
        return sorted(kept, key=lambda a: a["date"], reverse=True)

    def version(self) -> tuple[str, int, int]:
        """Changes whenever the file does (for caches)."""
        try:
            st = self._path.stat()
            return str(self._path), st.st_mtime_ns, st.st_size
        except OSError:
            return str(self._path), 0, 0

    def updated_at(self) -> str | None:
        return self._read().get("updated_at")

    def merge(self, new: list[dict]) -> None:
        by_id = {a["id"]: a for a in self._read().get("items", [])}
        by_id.update({a["id"]: a for a in new})
        data = {"updated_at": _iso(int(datetime.now(timezone.utc).timestamp())), "items": list(by_id.values())}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)


def mentions(name: str, items: list[dict]) -> list[dict]:
    """Patches whose text names the item: whole words, any case, plural allowed; the matching line as snippet."""
    words = r"\s+".join(re.escape(w) for w in name.split())
    pat = re.compile(rf"(?<![A-Za-z]){words}(e?s)?(?![A-Za-z])", re.I)
    out = []
    for a in items:
        line = next((ln for ln in a["text"].splitlines() if pat.search(ln)), None)
        if line:
            out.append({k: a[k] for k in ("id", "date", "title", "url")} | {"snippet": line.lstrip("- ").strip()})
    return sorted(out, key=lambda x: x["date"], reverse=True)


_CACHE: dict = {"version": None, "text": "", "items": [], "found": {}}


def cached_mentions(name: str, store: AnnouncementStore) -> list[dict]:
    """mentions() remembered until the stored patches change; a plain substring check over all patch text first,
    so the many items no patch names cost almost nothing (the overview asks for every item every minute)."""
    v = store.version()
    if _CACHE["version"] != v:
        items = store.items()
        _CACHE.update(version=v, items=items, found={}, text="\n".join(a["text"] for a in items).lower())
    if name not in _CACHE["found"]:
        hit = all(w in _CACHE["text"] for w in name.lower().split())
        _CACHE["found"][name] = mentions(name, _CACHE["items"]) if hit else []
    return _CACHE["found"][name]


def refresh(store: AnnouncementStore, get) -> str | None:
    """Fetch posts newer than the newest stored; returns an error message or None."""
    newest = store.items()[0]["date"] if store.items() else None
    since = int(datetime.fromisoformat(newest.replace("Z", "+00:00")).timestamp()) if newest else None
    try:
        store.merge(fetch_steam(get, since))
        return None
    except Exception as exc:   # offline / Steam down: keep the cache
        return f"公告暂时无法更新（{type(exc).__name__}）"


def due(store: AnnouncementStore) -> bool:
    u = store.updated_at()
    if not u:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(u.replace("Z", "+00:00"))
    return age.total_seconds() > REFRESH_HOURS * 3600
