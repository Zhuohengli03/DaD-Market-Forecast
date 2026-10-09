import json

from fastapi import APIRouter, Request

from ..schemas import ItemsConfigIn, ok

router = APIRouter(tags=["config"])


@router.get("/config/items")
def get_items_config(request: Request):
    path = request.app.state.settings.items_config_path
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": [], "categories": {}}
    return ok(data)


@router.put("/config/items")
def put_items_config(body: ItemsConfigIn, request: Request):
    path = request.app.state.settings.items_config_path
    path.write_text(json.dumps(body.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return ok(body.model_dump())
