from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.darkerdb_client import DarkerDBClient, DarkerDBError
from ..schemas import ok
from ..services.envfile import upsert_env
from ..data.app_settings import DARKERDB_LIVE_RATE_PER_MINUTE, MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS
from ..data.repository import make_repository

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


def environment(st) -> dict:
    import sys
    v = sys.version_info
    return {"python_version": f"{v.major}.{v.minor}.{v.micro}", "python_ok": (v.major, v.minor) >= (3, 10),
            "storage": st.repo.source_name(), "data_dir": str(st.settings.data_dir),
            "db_host": st.settings.db_host if st.settings.postgres_configured else None,
            "db_database": st.settings.db_database if st.settings.postgres_configured else None}


def probe_postgres(**dsn) -> None:
    import psycopg2
    with psycopg2.connect(connect_timeout=3, **dsn) as c, c.cursor() as cur:
        cur.execute("SELECT 1")


def setup_state(request: Request) -> dict:
    st = request.app.state
    key = st.settings.darker_market_api_key
    prefs = st.app_settings.get()
    return {
        "initialized": bool(prefs["initialized_at"]), "collect_interval_seconds": prefs["collect_interval_seconds"],
        "rate_limit_per_minute": DARKERDB_LIVE_RATE_PER_MINUTE,
        "auto_collect": prefs["auto_collect"], "environment": environment(st),
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


class ScheduleIn(BaseModel):
    interval_seconds: int
    auto_collect: bool


@router.put("/setup/schedule")
def put_schedule(body: ScheduleIn, request: Request):
    if not MIN_INTERVAL_SECONDS <= body.interval_seconds <= MAX_INTERVAL_SECONDS:
        raise HTTPException(422, f"采集间隔需要在 {MIN_INTERVAL_SECONDS} 秒到 30 天之间 / interval must be between "
                                 f"{MIN_INTERVAL_SECONDS} seconds and 30 days")
    st = request.app.state
    st.app_settings.update(collect_interval_seconds=body.interval_seconds, auto_collect=body.auto_collect)
    if body.auto_collect:
        st.scheduler.restart_if_running()
        if not st.scheduler.running():
            try:
                st.scheduler.start()
            except FileNotFoundError as exc:
                raise HTTPException(500, f"找不到定时采集程序 / scheduler script missing: {exc}")
    else:
        st.scheduler.stop()
    return ok(setup_state(request))


class DatabaseIn(BaseModel):
    host: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+$")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(min_length=1, max_length=63, pattern=r"^[A-Za-z0-9_]+$")
    user: str = Field(min_length=1, max_length=63, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(default="", max_length=256, pattern=r"^[^\r\n]*$")


def _use_repo(st) -> None:
    st.repo = make_repository(st.settings, log=lambda s: None)
    st.item_stats.invalidate()


@router.put("/setup/database")
def put_database(body: DatabaseIn, request: Request):
    st = request.app.state
    try:
        probe_postgres(host=body.host, port=body.port, dbname=body.database, user=body.user, password=body.password)
    except Exception:
        raise HTTPException(422, "连接不上这个 Postgres 数据库，请检查地址、端口、库名、用户名和密码。 / "
                                 "Could not connect to this Postgres database; check host, port, database, user and password.")
    for k, v in (("DB_HOST", body.host), ("DB_PORT", str(body.port)), ("DB_DATABASE", body.database),
                 ("DB_USER", body.user), ("DB_PASSWORD", body.password)):
        upsert_env(st.settings.env_path, k, v)
    s = st.settings
    s.db_host, s.db_port, s.db_database, s.db_user, s.db_password = body.host, body.port, body.database, body.user, body.password
    _use_repo(st)
    return ok(setup_state(request))


@router.delete("/setup/database")
def delete_database(request: Request):
    st = request.app.state
    upsert_env(st.settings.env_path, "DB_USER", "")
    st.settings.db_user = ""
    _use_repo(st)
    return ok(setup_state(request))


class InitIn(BaseModel):
    item_ids: list[str] = Field(max_length=3000)
    max_pages: int = Field(default=200, ge=1, le=600)


@router.post("/setup/initialize", status_code=202)
def initialize(body: InitIn, request: Request):
    from ..services.initializer import run_initialization
    st = request.app.state
    if not st.settings.darker_market_api_key:
        raise HTTPException(409, "请先填写 DarkerDB API key / set your DarkerDB API key first")
    if not body.item_ids:
        raise HTTPException(422, "请至少选择一个道具 / choose at least one item")
    if st.jobs.find_active("init"):
        raise HTTPException(409, "初始化正在进行中 / initialization already running")
    job = st.jobs.create("init", "首次初始化 / first-run setup")
    st.jobs.run_in_thread(job, lambda j: run_initialization(st, j, body.item_ids, body.max_pages))
    return ok({"job_id": job.id})
