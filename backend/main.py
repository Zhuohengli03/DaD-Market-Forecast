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


class _NoJobs:  # placeholder until services/jobs.py exists (removed in Task 9)
    def active(self):
        return []

    def shutdown(self):
        pass


class _NoScheduler:  # placeholder until services/scheduler_ctl.py exists (removed in Task 9)
    def running(self):
        return False


def create_app(settings: Settings | None = None, repo=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .data.repository import make_repository
        app.state.settings = settings
        app.state.repo = repo or make_repository(settings)
        _attach_services(app)
        yield
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
            "items": [i.__dict__ for i in st.repo.list_items()],
            "jobs_running": [j.to_dict() for j in st.jobs.active()],
            "scheduler_running": st.scheduler.running(),
        })

    _include_routers(app)
    _mount_frontend(app, settings)
    return app


def _attach_services(app: FastAPI) -> None:
    st = app.state
    try:
        from .data.seasons import SeasonStore
        st.seasons = SeasonStore(st.settings.seasons_path)
    except ImportError:
        st.seasons = None
    try:
        from .services.forecast_runs import RunStore
        st.runs = RunStore(st.settings.runs_dir)
    except ImportError:
        st.runs = None
    try:
        from .services.jobs import JobManager
        st.jobs = JobManager()
    except ImportError:
        st.jobs = _NoJobs()
    try:
        from .services.scheduler_ctl import SchedulerController
        st.scheduler = SchedulerController(st.settings, st.jobs)
    except ImportError:
        st.scheduler = _NoScheduler()


def _include_routers(app: FastAPI) -> None:
    for name in ROUTER_MODULES:
        try:
            module = importlib.import_module(f".api.{name}", __package__)
        except ImportError:
            continue
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
        candidate = dist / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")
