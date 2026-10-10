import json

from backend.services.collection import collector_argv


def test_collector_argv_for_catalog_item(settings):
    settings.bootstrap()
    argv = collector_argv(settings, {"name": "Arming Sword", "item_id": "id.item.arming_sword_2001"})
    assert argv[1:] == ["-m", "src.api.collect", "--item", "Arming Sword", "--item-id", "id.item.arming_sword_2001"]


def test_collector_argv_rejects_script_outside_src_api(settings):
    settings.bootstrap()
    import pytest
    with pytest.raises(ValueError):
        collector_argv(settings, {"name": "X", "file": "../../evil.py"})


def test_scheduler_tasks_follow_tracked_items(client, project_root):
    cfg = json.loads((project_root / "data" / "items.json").read_text())
    cfg["items"].append({"name": "Gold Ore", "item_id": "id.item.gold_ore_1001", "enabled": True, "schedule": "06:30"})
    cfg["items"].append({"name": "Off", "item_id": "id.item.off_1001", "enabled": False})
    client.put("/api/config/items", json=cfg)
    tasks = client.get("/api/scheduler").json()["data"]["tasks"]
    names = {t["item"]: t for t in tasks}
    assert set(names) == {"Test Ore", "Gold Ore"}
    assert names["Gold Ore"]["schedule_value"] == "06:30" and names["Test Ore"]["schedule_value"] == "23:00"
