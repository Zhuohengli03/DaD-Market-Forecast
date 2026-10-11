"""Rules that need upkeep: season market openings and hand-over times, which Steam posts count as patches.

They ship in src/rules_default.json. A newer, valid data/rules.json (synced from a rules service, or edited by
hand) takes over; anything older, unreadable or malformed is ignored, so a bad sync never breaks the app.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_BUNDLED = Path(__file__).with_name("rules_default.json")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_override_dir: Path | None = None
_cache: dict = {"key": None, "rules": None}


def set_data_dir(path: Path) -> None:
    """Where data/rules.json lives (the app's data directory)."""
    global _override_dir
    _override_dir = Path(path)
    clear_cache()


def _data_dir() -> Path:
    if _override_dir is not None:
        return _override_dir
    return Path(os.getenv("DARKER_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")


def clear_cache() -> None:
    _cache.update(key=None, rules=None)


def bundled() -> dict:
    return json.loads(_BUNDLED.read_text(encoding="utf-8"))


def validate(r: dict) -> dict:
    """The rules, or ValueError naming the first thing that is wrong."""
    if not isinstance(r, dict) or not isinstance(r.get("version"), int):
        raise ValueError("version must be a whole number")
    seasons, ann = r.get("seasons"), r.get("announcements")
    if not isinstance(seasons, dict) or not isinstance(ann, dict):
        raise ValueError("seasons and announcements are required")
    for sid, pair in (seasons.get("market_openings") or {}).items():
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(d, str) and _DATE.match(d) for d in pair)):
            raise ValueError(f"market_openings.{sid} must be [season start date, market open date]")
    for sid, b in (seasons.get("boundaries") or {}).items():
        if not isinstance(b, dict) or not set(b) <= {"start", "end"}:
            raise ValueError(f"boundaries.{sid} may only set start / end")
        for v in b.values():
            datetime.fromisoformat(v.replace("Z", "+00:00"))
    for k in ("patch", "not_patch"):
        if not all(isinstance(w, str) and w for w in ann.get(k, [])):
            raise ValueError(f"announcements.{k} must be a list of words")
    return r


def current() -> dict:
    """The rules in force, with "source": "bundled" or "synced"."""
    path = _data_dir() / "rules.json"
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(path), 0, 0)
    if _cache["key"] == key:
        return _cache["rules"]
    base = {**bundled(), "source": "bundled"}
    chosen = base
    if key[1]:
        try:
            synced = validate(json.loads(path.read_text(encoding="utf-8")))
            if synced["version"] >= base["version"]:
                chosen = {**synced, "source": "synced"}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    _cache.update(key=key, rules=chosen)
    return chosen


def sync(url: str, key: str, get) -> str | None:
    """Fetch rules from a rules service (`get(url, headers) -> dict`), keep them when valid; error text or None."""
    try:
        new = validate(get(url, {"X-License-Key": key, "Accept": "application/json"}))
    except ValueError as exc:
        return f"规则服务返回的内容不对：{exc}"
    except Exception as exc:   # offline, refused, bad key: keep what we have
        return f"规则暂时无法同步（{type(exc).__name__}）"
    path = _data_dir() / "rules.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**new, "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    clear_cache()
    return None
