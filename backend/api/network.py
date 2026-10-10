"""LAN access and port, set from Settings instead of editing .env; applied by restarting from the app."""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..schemas import ok
from ..services.envfile import upsert_env

router = APIRouter(tags=["network"])
logger = logging.getLogger(__name__)
LAN_HOST, LOCAL_HOST = "0.0.0.0", "127.0.0.1"
MIN_PORT, MAX_PORT = 1024, 65535


def _saved(settings) -> dict[str, str]:
    """HOST/PORT as saved in .env (what the next start will use)."""
    out: dict[str, str] = {}
    path = settings.env_path
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() in ("HOST", "PORT"):
                out[k.strip()] = v.strip().strip('"\'')
    return out


def lan_ips() -> list[str]:
    """This machine's address on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return [] if ip.startswith("127.") else [ip]
    except OSError:
        return []


def state(settings) -> dict:
    saved = _saved(settings)
    host = saved.get("HOST", settings.host)
    port = int(saved.get("PORT", settings.port))
    lan = host == LAN_HOST
    return {"lan": lan, "port": port, "running_port": settings.port, "running_lan": settings.host == LAN_HOST,
            "restart_required": (lan, port) != (settings.host == LAN_HOST, settings.port),
            "lan_urls": [f"http://{ip}:{port}" for ip in lan_ips()] if lan else []}


@router.get("/setup/network")
def get_network(request: Request):
    return ok(state(request.app.state.settings))


class NetworkIn(BaseModel):
    lan: bool
    port: int


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


@router.put("/setup/network")
def put_network(body: NetworkIn, request: Request):
    settings = request.app.state.settings
    if not MIN_PORT <= body.port <= MAX_PORT:
        raise HTTPException(422, f"端口需要在 {MIN_PORT} 到 {MAX_PORT} 之间 / port must be between {MIN_PORT} and {MAX_PORT}")
    if body.port != settings.port and not _port_free(body.port):
        raise HTTPException(409, f"端口 {body.port} 已被其他程序占用，请换一个 / port {body.port} is already in use")
    upsert_env(settings.env_path, "HOST", LAN_HOST if body.lan else LOCAL_HOST)
    upsert_env(settings.env_path, "PORT", str(body.port))
    return ok(state(settings))


def default_restarter(app) -> None:
    """Stop background work, then replace this process with a fresh server that reads the new settings."""
    try:
        app.state.scheduler.stop()
        app.state.jobs.shutdown()
    except Exception:  # noqa: BLE001 - restarting matters more than a clean stop
        logger.warning("stopping background work before restart failed", exc_info=True)
    env = {k: v for k, v in os.environ.items() if k not in ("HOST", "PORT")}   # let .env decide
    os.execve(sys.executable, [sys.executable, "-m", "backend"], env)


@router.post("/setup/restart", status_code=202)
def restart(request: Request):
    """Restart the app so new network settings apply; the page reconnects by itself."""
    app = request.app
    restarter = getattr(app.state, "restarter", None) or (lambda: default_restarter(app))

    def later():
        time.sleep(0.5)   # let this response reach the browser first
        restarter()
    threading.Thread(target=later, daemon=True, name="restart").start()
    return ok({"restarting": True})
