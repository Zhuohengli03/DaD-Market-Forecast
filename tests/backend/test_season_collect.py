"""Season boundaries at their announced times, and collecting one chosen season from the Collect page."""
import sys
import time

import pandas as pd

from backend.schemas import SeasonIn
from src.season_calendar import with_market_opening

H = {"X-Darker-Market": "1"}


def _wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_season_10_ends_at_the_announced_hour_not_midnight():
    # Hotfix #126: Season 10 ends on October 9 at 3:00 AM UTC; imports only carry the date
    s10 = with_market_opening({"id": "early-access-season-10", "start": "2026-07-17T00:00:00+00:00", "end": "2026-10-09T00:00:00+00:00"})
    s11 = with_market_opening({"id": "early-access-season-11", "start": "2026-10-09T00:00:00+00:00", "end": None})
    assert s10["end"] == "2026-10-09T03:00:00+00:00" and s11["start"] == "2026-10-09T03:00:00+00:00"
    assert s11["market_open"] == "2026-10-23"
    # a time the user set by hand is kept
    assert with_market_opening({"id": "early-access-season-10", "start": "2026-07-17T00:00:00+00:00", "end": "2026-10-09T05:00:00+00:00"})["end"] == "2026-10-09T05:00:00+00:00"


def _season(client, sid):
    return next(s for s in client.app.state.seasons.list() if s["id"] == sid)


def test_collect_all_for_a_chosen_season(client, project_root):
    st = client.app.state
    seen = []
    script = project_root / "fake.py"
    script.write_text('import json\nprint("@@" + json.dumps({"stage": "done", "new": 1}), flush=True)\n')
    st.collector_argv = lambda entry, since, until=None: (seen.append((since, until)), [sys.executable, str(script)])[1]
    r = client.post("/api/collect-all", json={"season_id": "s1"}, headers=H)
    assert r.status_code == 202
    _wait(client, r.json()["data"]["job_id"])
    s1 = _season(client, "s1")
    assert seen == [(s1["start"], s1["end"])]
    est = client.get("/api/estimate/collect-all", params={"season_id": "s1"}).json()["data"]
    assert est["items"] == 1 and est["since"] == s1["start"] and est["until"] == s1["end"]


def test_no_collection_before_the_market_opens(client):
    st = client.app.state
    s = _season(client, "s2")
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    st.seasons.update("s2", SeasonIn(name=s["name"], start="2026-01-20T00:00:00+00:00", end=None, note="", market_open=future))
    r = client.post("/api/collect/Test Ore", params={"season_id": "s2"}, headers=H)
    assert r.status_code == 409 and future in r.json()["error"]
    r = client.post("/api/collect-all", json={"season_id": "s2"}, headers=H)
    assert r.status_code == 409 and future in r.json()["error"]
    est = client.get("/api/estimate/collect-all", params={"season_id": "s2"}).json()["data"]
    assert est["requests"] == 0 and est["opens"] == future


def test_single_item_estimate_says_when_the_market_opens(client):
    st = client.app.state
    s = _season(client, "s2")
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    st.seasons.update("s2", SeasonIn(name=s["name"], start="2026-01-20T00:00:00+00:00", end=None, note="", market_open=future))
    d = client.get("/api/estimate/collect/Test Ore", params={"season_id": "s2"}).json()["data"]
    assert d["requests"] == 0 and d["opens"] == future
    assert client.get("/api/estimate/collect/Test Ore", params={"season_id": "s1"}).json()["data"]["opens"] is None


def test_a_finished_season_is_collected_from_its_start_and_continued(client, settings):
    import json
    st = client.app.state
    s1 = _season(client, "s1")                                   # ended season, nothing stored in it
    from backend.services.collection import collect_range, season_progress
    assert collect_range(st, "Test Ore", "s1") == (s1["start"], s1["end"])
    (settings.data_dir / "coverage").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "coverage" / "test_ore.json").write_text(json.dumps({"s1": "2025-11-15T00:00:00Z"}))
    assert collect_range(st, "Test Ore", "s1") == ("2025-11-15T00:00:00Z", s1["end"])
    p = season_progress(st, "Test Ore", "s1")
    assert p["through"] == "2025-11-15T00:00:00Z" and p["complete"] is False and 0 < p["share"] < 1
    (settings.data_dir / "coverage" / "test_ore.json").write_text(json.dumps({"s1": s1["end"]}))
    assert season_progress(st, "Test Ore", "s1")["complete"] is True
    r = client.post("/api/collect/Test Ore", params={"season_id": "s1"}, headers=H)
    assert r.status_code == 409 and "已经采集完整" in r.json()["error"]
    d = client.get("/api/estimate/collect/Test Ore", params={"season_id": "s1"}).json()["data"]
    assert d["requests"] == 0 and d["progress"]["complete"] is True
