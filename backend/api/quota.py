"""DarkerDB plan credits, as recorded by the client after each request (no request is made here)."""
import json
import logging

from fastapi import APIRouter, Request

from ..schemas import ok

router = APIRouter(tags=["quota"])
logger = logging.getLogger(__name__)
EMPTY = {"credits_remaining": None, "rate_limit": None, "updated_at": None}


def read_quota(st) -> dict:
    path = st.settings.data_dir / "quota.json"
    data = dict(EMPTY)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            data.update({k: raw.get(k) for k in EMPTY})
        except (OSError, ValueError):
            logger.warning("unreadable quota file %s", path, exc_info=True)
    return {**data, "monthly_quota": st.app_settings.get()["monthly_quota"]}


@router.get("/quota")
def get_quota(request: Request):
    return ok(read_quota(request.app.state))
