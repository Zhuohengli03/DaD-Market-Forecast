from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient, DarkerDBError
from ..schemas import ok
from ..services.envfile import upsert_env

router = APIRouter(tags=["setup"])


class ApiKeyIn(BaseModel):
    api_key: str = Field(min_length=1, max_length=256, pattern=r"^\s*[A-Za-z0-9._~+/=-]+\s*$")


def key_hint(key: str) -> str | None:
    return f"…{key[-4:]}" if key else None


def default_key_validator(key: str) -> None:
    """Raises DarkerDBError if DarkerDB rejects the key."""
    client = DarkerDBClient(key, min_interval=0, max_retries=1, timeout=15)
    client.get("/items", {"limit": 1})   # darkerdb.data scope
    client.get("/market", {"limit": 1})  # darkerdb.live scope


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
            raise HTTPException(422, "DarkerDB 拒绝了这个 key：请确认它是 v2 key，并勾选了 darkerdb.live 和 darkerdb.data 两个权限。 / "
                                     "DarkerDB rejected this key: use a v2 key with both the darkerdb.live and darkerdb.data scopes.")
        raise HTTPException(502, f"DarkerDB 暂时无法验证 key（HTTP {exc.code}），请稍后再试。 / "
                                 f"DarkerDB could not verify the key right now (HTTP {exc.code}); try again later.")
    except Exception:
        raise HTTPException(502, "连接 DarkerDB 失败，请检查网络后重试。 / Could not reach DarkerDB; check your network connection and retry.")
    settings = request.app.state.settings
    upsert_env(settings.env_path, "DARKER_MARKET_API_KEY", key)
    settings.darker_market_api_key = key
    return ok(setup_state(request))
