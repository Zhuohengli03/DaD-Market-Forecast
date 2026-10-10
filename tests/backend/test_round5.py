import json
import sys
import time

import pytest

from backend.services.catalog_sync import check_tradability
from backend.services.equipment import build_market_params
from src.scheduler.task_config import tasks_from_items

H = {"X-Darker-Market": "1"}


def _wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ---- 1. per-item schedules ---------------------------------------------------
def test_per_item_schedule_overrides_the_default(tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [
        {"name": "Iron Ore", "item_id": "id.item.iron_ores_2001", "enabled": True},
        {"name": "Gold Ore", "item_id": "id.item.gold_ores_5001", "enabled": True, "schedule": {"mode": "interval", "seconds": 45}},
        {"name": "Copper Ore", "enabled": True, "schedule": {"mode": "daily", "times": ["03:00", "15:30"]}},
        {"name": "Legacy", "enabled": True, "schedule": "23:00"},
    ]}))
    (tmp_path / "settings.json").write_text(json.dumps({"collect_interval_seconds": 600}))
    t = {x["item"]: x for x in tasks_from_items(items, settings_path=tmp_path / "settings.json")}
    assert (t["Iron Ore"]["schedule_type"], t["Iron Ore"]["schedule_value"]) == ("seconds", 600)
    assert (t["Gold Ore"]["schedule_type"], t["Gold Ore"]["schedule_value"]) == ("seconds", 45)
    assert (t["Copper Ore"]["schedule_type"], t["Copper Ore"]["schedule_value"]) == ("daily_times", ["03:00", "15:30"])
    assert t["Copper Ore"]["description"] == "每天 03:00、15:30 采集 Copper Ore"
    assert (t["Legacy"]["schedule_type"], t["Legacy"]["schedule_value"]) == ("daily_times", ["23:00"])


def test_item_schedule_endpoint_validates(client):
    r = client.put("/api/items/Test Ore/schedule", json={"mode": "interval", "seconds": 10}, headers=H)
    assert r.status_code == 422
    r = client.put("/api/items/Test Ore/schedule", json={"mode": "daily", "times": ["25:00"]}, headers=H)
    assert r.status_code == 422
    r = client.put("/api/items/Test Ore/schedule", json={"mode": "daily", "times": ["15:30", "03:00", "03:00"]}, headers=H)
    assert r.status_code == 200
    task = client.get("/api/scheduler").json()["data"]["tasks"][0]
    assert task["schedule_type"] == "daily_times" and task["schedule_value"] == "03:00、15:30"
    assert client.put("/api/items/Test Ore/schedule", json={"mode": "default"}, headers=H).status_code == 200
    assert client.put("/api/items/Ghost/schedule", json={"mode": "default"}, headers=H).status_code == 404


def test_core_scheduler_registers_several_daily_times(tmp_path):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src" / "scheduler"))
    import core_scheduler
    import schedule
    schedule.clear()
    s = core_scheduler.SmartScheduler.__new__(core_scheduler.SmartScheduler)
    s.running_tasks, s._queued = {}, set()
    import threading
    s._queued_lock = threading.Lock()
    script = tmp_path / "x.py"; script.write_text("")
    s.add_task_from_config({"name": "C", "script_path": str(script), "args": [], "schedule_type": "daily_times",
                            "schedule_value": ["03:00", "15:30"], "enabled": True})
    assert sorted(str(j.at_time) for j in schedule.get_jobs()) == ["03:00:00", "15:30:00"]
    schedule.clear()


# ---- 2. equipment sort ---------------------------------------------------------
def test_equipment_sort_is_passed_to_darkerdb():
    assert build_market_params("id.item.a", None, [], False, None, sort="price_per_unit:asc")["sort"] == "price_per_unit:asc"
    assert build_market_params("id.item.a", None, [], False, None)["sort"] == "created_at:desc"


def test_equipment_sort_is_validated(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    _seed(settings)
    client.app.state.settings.darker_market_api_key = "k"
    body = {"archetype": "id.item.arming_sword", "sort": "price:drop table"}
    assert client.post("/api/equipment/query", json=body, headers=H).status_code == 422


# ---- 3. season-aware listing counts ----------------------------------------------
def test_trade_check_records_first_listing_and_season_count(settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    calls = []

    class C:
        def get(self, path, params=None):
            calls.append(dict(params))
            if params.get("from"):
                return {"body": [{"created_at": "2026-10-09T03:00:00Z"}], "pagination": {"total": 12}}
            return {"body": [{"created_at": "2024-09-06T10:00:00.000Z"}], "pagination": {"total": 900}}
    check_tradability(C(), store, log=lambda s: None, season_start="2026-10-09T00:00:00+00:00")
    info = store.read()["trade_info"]["id.item.arming_sword"]
    assert info == {"total": 900, "first": "2024-09-06T10:00:00Z", "season": 12, "season_start": "2026-10-09T00:00:00+00:00"}
    assert {"archetype": "id.item.arming_sword", "limit": 1, "sort": "created_at:asc"} in calls
    rows, _, _ = store.query()
    sword = next(r for r in rows if r["name"] == "Arming Sword")
    assert sword["listings_total"] == 900 and sword["first_listed"] == "2024-09-06T10:00:00Z" and sword["season_listings"] == 12


def test_new_season_recounts_only_the_season_numbers(settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    n = {"calls": 0}

    class C:
        def get(self, path, params=None):
            n["calls"] += 1
            return {"body": [{"created_at": "2024-09-06T10:00:00Z"}], "pagination": {"total": 5}}
    check_tradability(C(), store, log=lambda s: None, season_start="2026-07-17T00:00:00+00:00")
    first_round = n["calls"]
    check_tradability(C(), store, log=lambda s: None, season_start="2026-07-17T00:00:00+00:00")
    assert n["calls"] == first_round                          # nothing new to check
    check_tradability(C(), store, log=lambda s: None, season_start="2026-10-09T00:00:00+00:00")
    assert n["calls"] == first_round + len(store.types())     # one season count per type, history kept


# ---- 4. add and collect many items from the Collect page ---------------------------
def test_bulk_track_skips_equipment_and_duplicates(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    _seed(settings)
    r = client.post("/api/items/track-bulk", json={"item_ids": ["id.item.bandage_1001", "id.item.bandage_2001",
                                                                "id.item.arming_sword_6001", "id.item.bandage_1001", "id.item.nope"]}, headers=H)
    d = r.json()["data"]
    assert r.status_code == 200 and d["added"] == 2 and d["skipped"] == {"has_stats": 1, "already_tracked": 1, "not_in_catalog": 1}


def test_collect_all_runs_every_tracked_item_in_one_job(client, project_root):
    script = project_root / "fake.py"
    script.write_text('import json\nprint("@@" + json.dumps({"stage": "done", "new": 3}), flush=True)\n')
    client.app.state.collector_argv = lambda entry, since: [sys.executable, str(script)]
    r = client.post("/api/collect-all", headers=H)
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded" and job["progress"]["stage"] == "done"
    assert job["progress"]["detail"]["items"][0]["collected"] == 3
