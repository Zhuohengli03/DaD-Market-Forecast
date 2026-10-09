from fastapi import APIRouter, Request

from ..schemas import ItemsConfigIn, ok

router = APIRouter(tags=["config"])


@router.get("/config/items")
def get_items_config(request: Request):
    return ok(request.app.state.items.read())


@router.put("/config/items")
def put_items_config(body: ItemsConfigIn, request: Request):
    data = body.model_dump(exclude_none=True)
    request.app.state.items.write(data)
    request.app.state.item_stats.invalidate()
    return ok(data)
