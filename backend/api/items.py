import json

from fastapi import APIRouter, Request

from ..schemas import ok

router = APIRouter(tags=["items"])


@router.get("/items")
def list_items(request: Request):
    st = request.app.state
    cfg_path = st.settings.items_config_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"items": []}
    stats = {i.name: i for i in st.repo.list_items()}
    out = []
    for it in cfg.get("items", []):
        s = stats.get(it["name"])
        out.append({**it, "rows": s.rows if s else 0, "first_ts": s.first_ts if s else None,
                    "last_ts": s.last_ts if s else None})
    return ok(out)
