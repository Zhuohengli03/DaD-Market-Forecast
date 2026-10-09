from fastapi import APIRouter, HTTPException, Request

from ..schemas import ok
from ..services.collection import start_collection

router = APIRouter(tags=["collect"])


@router.post("/collect/{item}", status_code=202)
def collect(item: str, request: Request):
    try:
        job = start_collection(request.app.state, item)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return ok({"job_id": job.id})
