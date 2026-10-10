"""Request-cost estimates: always rounded up, so the user is never surprised by a bigger bill."""
import json
import sys

import pandas as pd
import pytest

from backend.services.cost import (CATALOG_ITEMS_GUESS, MAX_PAGES, collect_estimate, collect_all_estimate,
                                   items_estimate, schedule_estimate, sync_cost)

H = {"X-Darker-Market": "1"}


def _last(client):
    return pd.Timestamp(next(i for i in client.app.state.item_stats.get() if i.name == "Test Ore").last_ts)


def test_collect_estimate_scales_with_time_since_last_listing(client):
    st = client.app.state
    entry = st.items.get("Test Ore")
    last = _last(client)
    six = collect_estimate(st, entry, now=last + pd.Timedelta(hours=6))
    day = collect_estimate(st, entry, now=last + pd.Timedelta(days=3))
    # fixture: 5 listings an hour -> 30 new in 6 h; with the safety margin and a boundary page that is 2 pages
    assert six == {"requests": 2, "known": True}
    assert day["requests"] > six["requests"] and day["requests"] >= -(-5 * 72 // 50)
    capped = collect_estimate(st, entry, now=last + pd.Timedelta(days=400))
    assert capped["requests"] == MAX_PAGES + 1


def test_item_without_local_data_costs_up_to_its_season_or_the_page_cap(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    store = _seed(settings)
    st = client.app.state
    st.items.track("Iron Ore", "id.item.iron_ores_2001", "misc")
    assert collect_estimate(st, st.items.get("Iron Ore")) == {"requests": MAX_PAGES + 1, "known": False}
    store.save_trade_entry("id.item.iron_ores", {"total": 9000, "first": None, "season": 120, "season_start": "2026-01-20T00:00:00+00:00"})
    # 120 listings this season for the whole type (an upper bound for one rarity): 3 pages + 1
    assert collect_estimate(st, st.items.get("Iron Ore")) == {"requests": 4, "known": True}


def test_collect_all_and_picker_estimates_add_up(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    _seed(settings)
    st = client.app.state
    st.items.track("Bandage (poor)", "id.item.bandage_1001", "utility")
    total = collect_all_estimate(st)
    assert total["items"] == 2 and total["unknown"] == 1 and total["requests"] >= MAX_PAGES + 2
    est = items_estimate(st, ["id.item.bandage_1001", "id.item.bandage_2001"], max_pages=50)
    assert est == {"requests": 2 * 51, "unknown": 2, "items": {"id.item.bandage_1001": 51, "id.item.bandage_2001": 51}}


def test_schedule_estimate_uses_each_items_own_schedule(client):
    st = client.app.state
    st.items.set_schedule("Test Ore", {"mode": "daily", "times": ["08:00", "20:00"]})
    est = schedule_estimate(st, default_seconds=3600)
    assert est["per_item"]["Test Ore"]["runs"] == 60
    # 12 h between runs at 5/h with margin: 90 new -> 2 pages + 1
    assert est["per_item"]["Test Ore"]["per_run"] == 3 and est["per_month"] == 180


def test_sync_cost_counts_pages_in_both_languages(settings):
    from backend.data.catalog import CatalogStore
    settings.bootstrap()
    store = CatalogStore(settings.catalog_path)
    assert sync_cost(store) == 1 + 2 * -(-CATALOG_ITEMS_GUESS // 200)
    store.save_items([{"id": f"id.item.x_{i}", "archetype": "id.item.x", "name": "X"} for i in range(450)], "S11")
    assert sync_cost(store) == 1 + 2 * 3


def test_estimate_endpoints(client):
    r = client.get("/api/estimate/collect-all").json()["data"]
    assert r["items"] == 1 and r["requests"] >= 1
    r = client.get("/api/estimate/schedule", params={"interval_seconds": 3600}).json()["data"]
    assert r["per_month"] == r["per_item"]["Test Ore"]["runs"] * r["per_item"]["Test Ore"]["per_run"]
    r = client.post("/api/estimate/items", json={"item_ids": ["id.item.nope"], "max_pages": 200}, headers=H).json()["data"]
    assert r["requests"] == 201
    meta = client.get("/api/catalog/meta").json()["data"]
    assert meta["sync_cost"] >= 1 and "check_cost" in meta


def test_client_counts_credits_it_used():
    from tests.api.test_darkerdb_client import _Resp, _Session, _env
    from src.api.darkerdb_client import DarkerDBClient
    s = _Session([_Resp(200, _env([]), {"X-Credits-Cost": "2"}), _Resp(200, _env([]), {})])
    c = DarkerDBClient("k", session=s, min_interval=0)
    c.get("/a"); c.get("/b")
    assert c.credits_used == 3   # a response without the header counts as 1


def test_jobs_record_estimate_and_actual_credits(client, project_root):
    script = project_root / "fake.py"
    script.write_text('import json\nprint("@@" + json.dumps({"stage": "collect", "done": 1, "new": 2, "credits": 1}), flush=True)\n'
                      'print("@@" + json.dumps({"stage": "done", "new": 2, "credits": 3}), flush=True)\n')
    client.app.state.collector_argv = lambda entry, since: [sys.executable, str(script)]
    from tests.backend.test_round5 import _wait
    job = _wait(client, client.post("/api/collect-all", headers=H).json()["data"]["job_id"])
    assert job["cost"]["used"] == 3 and job["cost"]["estimate"] >= 1


def test_collect_all_estimate_lists_each_item(client):
    r = client.get("/api/estimate/collect-all").json()["data"]
    assert set(r["per_item"]) == {"Test Ore"} and r["per_item"]["Test Ore"] == r["requests"]


def test_one_catalog_type(client, settings):
    from tests.backend.test_types_and_equipment import _seed
    _seed(settings)
    d = client.get("/api/catalog/types/id.item.bandage").json()["data"]
    assert d["name"] == "Bandage" and len(d["variants"]) == 2 and "raw" in d["variants"][0]
    assert client.get("/api/catalog/types/id.item.nope").status_code == 404


def test_estimate_for_an_untracked_item_is_404(client):
    r = client.get("/api/estimate/collect/Nope Ore")
    assert r.status_code == 404 and "Nope Ore" in r.json()["error"]
    assert client.get("/api/estimate/collect/Test Ore").status_code == 200
