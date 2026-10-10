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
def test_trade_check_costs_one_request_per_type_and_records_first_listing(settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    calls = []

    class C:
        def get(self, path, params=None):
            calls.append(dict(params))
            return {"body": [{"created_at": "2024-09-06T10:00:00.000Z"}], "pagination": {"total": 900}}
    check_tradability(C(), store, log=lambda s: None)
    assert len(calls) == len(store.types())                  # one request per type, nothing for the season
    assert all(c == {"archetype": c["archetype"], "limit": 1, "sort": "created_at:asc"} for c in calls)
    info = store.read()["trade_info"]["id.item.arming_sword"]
    assert info["total"] == 900 and info["first"] == "2024-09-06T10:00:00Z" and info.get("season") is None
    sword = next(r for r in store.query()[0] if r["name"] == "Arming Sword")
    assert sword["listings_total"] == 900 and sword["first_listed"] == "2024-09-06T10:00:00Z"


def test_trade_check_reuses_earlier_totals_without_requests(settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings, totals={"id.item.arming_sword": 5, "id.item.iron_ores": 7, "id.item.bandage": 1, "id.item.gold_coins": 0})
    n = {"calls": 0}

    class C:
        def get(self, path, params=None):
            n["calls"] += 1
            return {"body": [], "pagination": {"total": 0}}
    check_tradability(C(), store, log=lambda s: None)
    assert n["calls"] == 0
    assert store.read()["trade_info"]["id.item.iron_ores"]["total"] == 7


def test_one_type_history_on_demand(settings):
    from backend.services.catalog_sync import type_history
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    calls = []

    class C:
        def get(self, path, params=None):
            calls.append(dict(params))
            if params.get("from"):
                return {"body": [{"created_at": "2026-10-09T03:00:00Z"}], "pagination": {"total": 12}}
            return {"body": [{"created_at": "2024-09-06T10:00:00Z"}], "pagination": {"total": 900}}
    season = "2026-10-09T00:00:00+00:00"
    info = type_history(C(), store, "id.item.arming_sword", season)
    assert info == {"total": 900, "first": "2024-09-06T10:00:00Z", "season": 12, "season_start": season}
    assert len(calls) == 2
    type_history(C(), store, "id.item.arming_sword", season)
    assert len(calls) == 2                                    # cached
    type_history(C(), store, "id.item.arming_sword", "2027-01-01T00:00:00+00:00")
    assert len(calls) == 3                                    # new season: only the season count again
    with pytest.raises(KeyError):
        type_history(C(), store, "id.item.nope", season)


def test_history_endpoint_uses_latest_season(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    _seed(settings)

    class C:
        def get(self, path, params=None):
            return {"body": [{"created_at": "2024-09-06T10:00:00Z"}], "pagination": {"total": 3 if params.get("from") else 50}}
    client.app.state.client_factory = lambda k: C()
    client.app.state.settings.darker_market_api_key = "k"
    r = client.post("/api/catalog/types/id.item.arming_sword/history", headers=H)
    d = r.json()["data"]
    assert r.status_code == 200 and d["total"] == 50 and d["season"] == 3 and d["season_start"].startswith("2026-01-20")   # latest season in the fixture
    assert client.post("/api/catalog/types/id.item.nope/history", headers=H).status_code == 404


def test_monthly_quota_setting(client):
    assert client.get("/api/setup").json()["data"]["monthly_quota"] == 10000
    r = client.put("/api/setup/quota", json={"monthly_quota": 50000}, headers=H)
    assert r.status_code == 200 and r.json()["data"]["monthly_quota"] == 50000
    assert client.put("/api/setup/quota", json={"monthly_quota": 10}, headers=H).status_code == 422


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


def test_collect_all_runs_items_in_parallel(client, project_root):
    from backend.services.initializer import PARALLEL_COLLECTS
    assert PARALLEL_COLLECTS == 3
    items = client.app.state.items
    for n in ("A Ore", "B Ore"):
        items.track(n, f"id.item.{n[0].lower()}_ore_1001", "ore")
    script = project_root / "slow.py"
    script.write_text('import json, time\ntime.sleep(1.0)\nprint("@@" + json.dumps({"stage": "done", "new": 1}), flush=True)\n')
    client.app.state.collector_argv = lambda entry, since: [sys.executable, str(script)]
    t0 = time.time()
    job = _wait(client, client.post("/api/collect-all", headers=H).json()["data"]["job_id"])
    took = time.time() - t0
    rows = job["progress"]["detail"]["items"]
    assert len(rows) == 3 and all(r["status"] == "collected" for r in rows)
    assert took < 2.5                                       # three 1-second collectors overlap


def test_items_list_has_icon_and_this_season_rows(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    data = store.read()
    data["items"] = [{**v, "icon_url": "https://x/iron.png"} if v["id"] == "id.item.iron_ores_2001" else v for v in data["items"]]
    store._write(data)
    client.app.state.items.track("Iron Ore", "id.item.iron_ores_2001", "misc")
    rows = {r["name"]: r for r in client.get("/api/items").json()["data"]}
    assert rows["Iron Ore"]["icon_url"] == "https://x/iron.png"
    test_ore = rows["Test Ore"]
    assert 0 < test_ore["season_rows"] <= test_ore["rows"]   # counted from the latest season's start, locally


def test_quota_endpoint_reads_what_the_client_recorded(client, settings):
    assert client.get("/api/quota").json()["data"]["credits_remaining"] is None
    (settings.data_dir / "quota.json").write_text(json.dumps({"credits_remaining": 7232, "rate_limit": 60, "updated_at": "2026-10-10T03:37:54Z"}))
    d = client.get("/api/quota").json()["data"]
    assert d == {"credits_remaining": 7232, "rate_limit": 60, "updated_at": "2026-10-10T03:37:54Z", "monthly_quota": 10000}


def test_status_lists_each_item_as_collecting_or_waiting(client, project_root, settings, monkeypatch):
    st = client.app.state
    for n in ("A Ore", "B Ore", "C Ore"):
        st.items.track(n, f"id.item.{n[0].lower()}_ore_1001", "ore")
    job = st.jobs.create("collect", "collect all", meta={"item": "*"})
    st.jobs.progress(job, "collect", 0, 3, items=[{"name": "A Ore", "status": "collecting"}, {"name": "B Ore", "status": "waiting"},
                                                 {"name": "Test Ore", "status": "collected"}])
    (settings.data_dir / "scheduler_state.json").write_text(json.dumps({"tasks": {"Collect C Ore": "waiting"}}))
    monkeypatch.setattr(st.scheduler, "running", lambda: True)
    monkeypatch.setattr(st.scheduler, "tasks", lambda: [{"name": "Collect C Ore", "item": "C Ore"}])
    monkeypatch.setattr(st.scheduler, "stop", lambda: None)
    d = client.get("/api/status").json()["data"]
    assert d["collecting"] == {"A Ore": "collecting", "B Ore": "waiting", "C Ore": "waiting"}


def test_untrack_many_at_once_keeps_collected_data(client):
    st = client.app.state
    for n in ("A Ore", "B Ore", "C Ore"):
        st.items.track(n, f"id.item.{n[0].lower()}_ore_1001", "ore")
    r = client.post("/api/items/untrack-bulk", json={"names": ["A Ore", "B Ore", "Nope"]}, headers=H)
    assert r.status_code == 200 and r.json()["data"] == {"removed": 2, "missing": 1}
    assert {i["name"] for i in st.items.list()} == {"Test Ore", "C Ore"}
    assert client.post("/api/items/untrack-bulk", json={"names": []}, headers=H).status_code == 422
