"""Settings → rules: which rules are in force, an optional rules service to sync from, and syncing now."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src import rules
from ..schemas import ok
from ..services.envfile import upsert_env

router = APIRouter(tags=["rules"])


def http_get(url: str, headers: dict) -> dict:
    import requests
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def _status(st) -> dict:
    r = rules.current()
    key = st.settings.rules_key or ""
    return {"source": r["source"], "version": r["version"], "updated": r.get("updated"), "synced_at": r.get("synced_at"),
            "url": st.app_settings.get()["rules_url"], "key_saved": bool(key), "key_hint": key[-4:] if key else None,
            "error": getattr(st, "rules_error", None)}


def sync_now(st) -> str | None:
    url = st.app_settings.get()["rules_url"]
    st.rules_error = rules.sync(url, st.settings.rules_key or "", st.rules_get) if url else None
    return st.rules_error


@router.get("/rules")
def status(request: Request):
    return ok(_status(request.app.state))


class SourceIn(BaseModel):
    url: str = Field("", max_length=500, pattern=r"^(https://\S+)?$")   # empty: maintain the rules yourself
    key: str | None = Field(None, max_length=200)


@router.put("/rules/source")
def set_source(body: SourceIn, request: Request):
    st = request.app.state
    st.app_settings.update(rules_url=body.url.strip())
    if body.key and body.key.strip():
        upsert_env(st.settings.env_path, "RULES_KEY", body.key.strip())
        st.settings.rules_key = body.key.strip()
    return ok(_status(st))


@router.post("/rules/sync")
def sync(request: Request):
    st = request.app.state
    if not st.app_settings.get()["rules_url"]:
        raise HTTPException(409, "还没有填写规则服务地址 / no rules service set")
    sync_now(st)
    return ok(_status(st))
