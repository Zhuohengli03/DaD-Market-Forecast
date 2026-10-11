import time


def _wait(client, job_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_season_holdout_job_roundtrip(client):
    r = client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s2", "models": ["seasonal_naive_24"]})
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded", job
    run_id = job["result_ref"]
    listed = client.get("/api/forecast/holdouts").json()["data"]
    assert listed[0]["run_id"] == run_id and listed[0]["item"] == "Test Ore" and "skill" in listed[0]
    full = client.get(f"/api/forecast/holdouts/{run_id}").json()["data"]
    assert full["train_hours"] == 140 and full["points"] and full["metrics"]["mae"] > 0
    assert client.get("/api/forecast/runs").json()["data"] == []          # holdouts are not forecast runs
    assert client.delete(f"/api/forecast/holdouts/{run_id}").status_code == 200
    assert client.get(f"/api/forecast/holdouts/{run_id}").status_code == 404


def test_season_holdout_validation(client):
    assert client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "nope"}).status_code == 404
    assert client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s2", "train_frac": 0.2}).status_code == 422
    assert client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s2", "models": ["ghost"]}).status_code == 422


def test_season_holdout_fails_cleanly_without_data(client):
    r = client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s1"})
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "failed" and "采集" in job["error"]


def test_holdout_at_4_hour_blocks(client):
    r = client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s2", "models": ["seasonal_naive_24"], "resolution": "4h"})
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded", job
    full = client.get(f"/api/forecast/holdouts/{job['result_ref']}").json()["data"]
    assert full["resolution"] == "4h" and "mdape" in full["metrics"]
    assert client.post("/api/forecast/holdout", json={"item": "Test Ore", "season_id": "s2", "resolution": "2h"}).status_code == 422
