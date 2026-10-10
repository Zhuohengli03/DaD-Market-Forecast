import time


def _wait(client, job_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_models_list(client):
    names = client.get("/api/forecast/models").json()["data"]
    assert "naive" in names and "ridge" in names


def test_forecast_job_roundtrip(client):
    r = client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s2", "horizon_hours": 24, "models": ["ridge"]})
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded", job
    run_id = job["result_ref"]
    runs = client.get("/api/forecast/runs").json()["data"]
    assert runs[0]["run_id"] == run_id and runs[0]["item"] == "Test Ore" and "beats_baseline" in runs[0]
    full = client.get(f"/api/forecast/runs/{run_id}").json()["data"]
    assert len(full["forecast"]) == 24
    actuals = client.get(f"/api/forecast/runs/{run_id}/actuals").json()["data"]
    assert actuals == []  # synthetic data ends where the forecast starts
    assert client.delete(f"/api/forecast/runs/{run_id}").status_code == 200
    assert client.get(f"/api/forecast/runs/{run_id}").status_code == 404


def test_forecast_validation(client):
    assert client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s2", "horizon_hours": 5}).status_code == 422
    assert client.post("/api/forecast", json={"item": "Test Ore", "season_id": "nope"}).status_code == 404
    assert client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s2", "models": ["ghost"]}).status_code == 422


def test_forecast_job_fails_cleanly_on_short_data(client):
    r = client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s1", "horizon_hours": 24, "models": ["ridge"]})
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "failed" and "连续" in job["error"]


def test_forecast_accepts_origin_step(client):
    r = client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s2", "horizon_hours": 24,
                                           "models": ["ridge"], "backtest_origin_step": 0})
    assert r.status_code == 422
    r = client.post("/api/forecast", json={"item": "Test Ore", "season_id": "s2", "horizon_hours": 24,
                                           "models": ["ridge"], "backtest_origin_step": 3})
    assert r.status_code == 202
    _wait(client, r.json()["data"]["job_id"])
