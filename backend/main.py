from __future__ import annotations

import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import fail, ok
from .settings import Settings

ROUTER_MODULES = ("items", "series", "listings", "seasons", "forecast", "jobs", "collect", "scheduler", "config")


def _format_validation(exc: RequestValidationError) -> str:
    parts = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        parts.append(f"{loc}: {e.get('msg')}")
    return "; ".join(parts) or "invalid request"



def create_app(settings: Settings | None = None, repo=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .data.repository import make_repository
        app.state.settings = settings
        app.state.repo = repo or make_repository(settings)
        _attach_services(app)
        yield
        app.state.scheduler.stop()
        app.state.jobs.shutdown()

    app = FastAPI(title="Darker Market", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content=fail(_format_validation(exc)))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content=fail(str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content=fail(f"{type(exc).__name__}: {exc}"))

    @app.get("/api/status")
    def status(request: Request):
        st = request.app.state
        return ok({
            "source": st.repo.source_name(),
            "python": settings.python_executable,
            "items": [i.__dict__ for i in st.item_stats.get()],
            "jobs_running": [j.to_dict() for j in st.jobs.active()],
            "scheduler_running": st.scheduler.running(),
        })

    _include_routers(app)
    _mount_frontend(app, settings)
    return app


def _attach_services(app: FastAPI) -> None:
    from .data.seasons import SeasonStore
    from .services.forecast_runs import RunStore
    from .services.jobs import JobManager
    from .services.scheduler_ctl import SchedulerController
    st = app.state
    st.seasons = SeasonStore(st.settings.seasons_path)
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
