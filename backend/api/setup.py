from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient, DarkerDBError
from ..schemas import ok
from ..services.envfile import upsert_env

router = APIRouter(tags=["setup"])


class ApiKeyIn(BaseModel):
    api_key: str = Field(min_length=1, max_length=256)


def key_hint(key: str) -> str | None:
    return f"…{key[-4:]}" if key else None


def default_key_validator(key: str) -> None:
    """Raises DarkerDBError if DarkerDB rejects the key."""
    DarkerDBClient(key, min_interval=0, max_retries=1, timeout=15).get("/items", {"limit": 1})


def setup_state(request: Request) -> dict:
    st = request.app.state
    key = st.settings.darker_market_api_key
    return {
        "api_key_configured": bool(key),
        "api_key_hint": key_hint(key),
        "data_source": st.repo.source_name(),
        "seasons": len(st.seasons.list()),
        "tracked_items": len(st.items.list()),
        "catalog_items": st.catalog.count(),
    }


@router.get("/setup")
def get_setup(request: Request):
    return ok(setup_state(request))


@router.put("/setup/api-key")
def put_api_key(body: ApiKeyIn, request: Request):
    key = body.api_key.strip()
    if not key:
        raise HTTPException(422, "API key 不能为空 / API key must not be empty")
    try:
        request.app.state.key_validator(key)
    except DarkerDBError as exc:
        if exc.code in (401, 403):
            raise HTTPException(422, f"DarkerDB 拒绝了这个 key（{exc}）。请确认它是 v2 key，并勾选了 darkerdb.live 和 darkerdb.data 权限。")
        raise HTTPException(502, f"DarkerDB 暂时无法验证 key：{exc}")
    except OSError as exc:
        raise HTTPException(502, f"连接 DarkerDB 失败，请检查网络后重试（{exc}）")
    settings = request.app.state.settings
    upsert_env(settings.env_path, "DARKER_MARKET_API_KEY", key)
    settings.darker_market_api_key = key
    return ok(setup_state(request))
