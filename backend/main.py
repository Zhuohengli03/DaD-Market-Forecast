from __future__ import annotations

import importlib
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import fail, ok
from .settings import Settings

log = logging.getLogger("darker_market")

ROUTER_MODULES = ("setup", "catalog", "equipment", "items", "series", "listings", "seasons", "forecast", "jobs", "collect", "scheduler", "config", "quota", "estimate", "overview", "network")


def _format_validation(exc: RequestValidationError) -> str:
    parts = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        parts.append(f"{loc}: {e.get('msg')}")
    return "; ".join(parts) or "invalid request"



from .services.collection import collection_states  # noqa: E402


def create_app(settings: Settings | None = None, repo=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .data.repository import make_repository
        settings.bootstrap()
        app.state.settings = settings
        app.state.repo = repo or make_repository(settings)
        _attach_services(app)
        if app.state.app_settings.get()["auto_collect"]:
            try:
                app.state.scheduler.start()  # resume scheduled collection chosen in the setup wizard
            except FileNotFoundError:
                pass
        yield
        app.state.scheduler.stop()
        app.state.jobs.shutdown()

    app = FastAPI(title="Darker Market", lifespan=lifespan)
    _add_guards(app, settings)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content=fail(_format_validation(exc)))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content=fail(str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        ref = uuid.uuid4().hex[:8]
        log.exception("unhandled error ref=%s", ref)
        return JSONResponse(status_code=500, content=fail(
            f"服务器内部错误，请查看后端终端日志（ref {ref}） / Internal error, see the backend log (ref {ref})"))

    @app.get("/api/status")
    def status(request: Request):
        st = request.app.state
        return ok({
            "source": st.repo.source_name(),
            "python": settings.python_executable,
            "items": [i.__dict__ for i in st.item_stats.get()],
            "jobs_running": [j.to_dict() for j in st.jobs.active()],
            "scheduler_running": st.scheduler.running(),
            "collecting": collection_states(st),
        })

    _include_routers(app)
    _mount_frontend(app, settings)
    return app


APP_HEADER = "X-Darker-Market"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
LOOPBACK_HOSTS = ["localhost", "127.0.0.1", "[::1]", "testserver"]


def _add_guards(app: FastAPI, settings: Settings) -> None:
    """Block cross-site writes and DNS rebinding.

    Every state-changing /api request must carry X-Darker-Market. Browsers only let other sites send a custom
    header after a CORS preflight, and this server answers no preflight, so web pages elsewhere cannot trigger
    jobs or change settings. The Host check stops DNS-rebinding pages from posing as localhost.
    """
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    @app.middleware("http")
    async def require_app_header(request: Request, call_next):
        if request.method not in SAFE_METHODS and request.url.path.startswith("/api/") \
                and request.headers.get(APP_HEADER) != "1":
            return JSONResponse(status_code=403, content=fail(
                "跨站请求已拒绝 / cross-site request refused (missing X-Darker-Market header)"))
        return await call_next(request)

    if settings.host in ("127.0.0.1", "localhost", "::1"):
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOOPBACK_HOSTS)


def _attach_services(app: FastAPI) -> None:
    from .api.setup import default_key_validator
    from .data.catalog import CatalogStore
    from .data.app_settings import AppSettings
    from .data.items import ItemsStore
    from .data.seasons import SeasonStore
    from .services.forecast_runs import RunStore
    from .services.jobs import JobManager
    from .services.scheduler_ctl import SchedulerController
    st = app.state
    st.seasons = SeasonStore(st.settings.seasons_path)
    st.items = ItemsStore(st.settings.items_config_path)
    st.app_settings = AppSettings(st.settings.app_settings_path)
    from .services.collection import collector_argv
    st.collector_argv = lambda entry, since: collector_argv(st.settings, entry) + (["--since", since] if since else [])
    st.catalog = CatalogStore(st.settings.catalog_path)
    st.key_validator = lambda k: default_key_validator(k, quota_file=st.settings.data_dir / "quota.json")
    st.runs = RunStore(st.settings.runs_dir)
    st.jobs = JobManager()
    st.scheduler = SchedulerController(st.settings, st.jobs)
    st.item_stats = _TtlCache(lambda: st.repo.list_items(), ttl=60.0)


class _TtlCache:
    """list_items() scans every table; /status is polled every few seconds."""

    def __init__(self, fn, ttl: float):
        self._fn, self._ttl, self._at, self._value = fn, ttl, 0.0, None

    def get(self):
        import time
        now = time.monotonic()
        if self._value is None or now - self._at > self._ttl:
            self._value, self._at = self._fn(), now
        return self._value

    def invalidate(self) -> None:
        self._value = None


def _include_routers(app: FastAPI) -> None:
    for name in ROUTER_MODULES:
        module = importlib.import_module(f".api.{name}", __package__)
        app.include_router(module.router, prefix="/api")


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    dist = settings.frontend_dist
    if dist is None or not (dist / "index.html").exists():
        return
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        root = dist.resolve()
        candidate = (root / path).resolve()
        if path and root in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(root / "index.html")
