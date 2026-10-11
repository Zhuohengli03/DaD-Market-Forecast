"""Pick which tracked items are collected on a schedule, one quality or a whole item type at a time."""
import json

from src.scheduler.task_config import tasks_from_items

H = {"X-Darker-Market": "1"}


def _track(st, *names):
    for n in names:
        st.items.track(n, f"id.item.{n.split()[0].lower()}_{abs(hash(n)) % 9999}", "misc")


def test_untick_items_from_the_schedule_in_bulk(client, settings):
    st = client.app.state
    _track(st, "Ruby (Cracked) (poor)", "Ruby (Flawed) (common)", "Gold Ore")
    r = client.put("/api/items/scheduled", json={"names": ["Ruby (Cracked) (poor)", "Ruby (Flawed) (common)", "Nope"], "scheduled": False}, headers=H)
    assert r.status_code == 200 and r.json()["data"] == {"changed": 2, "missing": 1}
    rows = {i["name"]: i for i in client.get("/api/items").json()["data"]}
    assert rows["Ruby (Cracked) (poor)"]["scheduled"] is False and rows["Gold Ore"].get("scheduled", True) is True
    # still tracked (shown, collectable by hand), just not on the schedule
    names = {t["item"] for t in tasks_from_items(settings.items_config_path, settings.project_root)}
    assert "Gold Ore" in names and "Ruby (Cracked) (poor)" not in names
    client.put("/api/items/scheduled", json={"names": ["Ruby (Cracked) (poor)"], "scheduled": True}, headers=H)
    names = {t["item"] for t in tasks_from_items(settings.items_config_path, settings.project_root)}
    assert "Ruby (Cracked) (poor)" in names


def test_monthly_estimate_counts_only_scheduled_items(client):
    st = client.app.state
    before = client.get("/api/estimate/schedule").json()["data"]
    assert "Test Ore" in before["per_item"]
    client.put("/api/items/scheduled", json={"names": ["Test Ore"], "scheduled": False}, headers=H)
    after = client.get("/api/estimate/schedule").json()["data"]
    assert "Test Ore" not in after["per_item"] and after["per_month"] == before["per_month"] - before["per_item"]["Test Ore"]["runs"] * before["per_item"]["Test Ore"]["per_run"]


def test_one_schedule_for_a_whole_item_type(client, settings):
    st = client.app.state
    _track(st, "Ruby (Cracked) (poor)", "Ruby (Flawed) (common)")
    r = client.put("/api/items/schedule-bulk", json={"names": ["Ruby (Cracked) (poor)", "Ruby (Flawed) (common)"],
                                                     "mode": "daily", "times": ["20:00", "08:00"]}, headers=H)
    assert r.status_code == 200 and r.json()["data"]["changed"] == 2
    items = {i["name"]: i for i in json.loads(settings.items_config_path.read_text())["items"]}
    assert items["Ruby (Flawed) (common)"]["schedule"] == {"mode": "daily", "times": ["08:00", "20:00"]}
    r = client.put("/api/items/schedule-bulk", json={"names": ["Ruby (Cracked) (poor)"], "mode": "default"}, headers=H)
    items = {i["name"]: i for i in json.loads(settings.items_config_path.read_text())["items"]}
    assert "schedule" not in items["Ruby (Cracked) (poor)"]
    assert client.put("/api/items/schedule-bulk", json={"names": ["Gold"], "mode": "interval", "seconds": 5}, headers=H).status_code == 422


def test_turning_the_schedule_on_or_off_is_remembered(client, monkeypatch):
    st = client.app.state
    monkeypatch.setattr(st.scheduler, "start", lambda: True)
    monkeypatch.setattr(st.scheduler, "stop", lambda: False)
    client.post("/api/scheduler/start", headers=H)
    assert st.app_settings.get()["auto_collect"] is True        # resumes after the app restarts
    client.post("/api/scheduler/stop", headers=H)
    assert st.app_settings.get()["auto_collect"] is False
