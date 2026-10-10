import json
import time


def _wait(client, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_collect_runs_script(client):
    r = client.post("/api/collect/Test Ore")
    assert r.status_code == 202
    job = _wait(client, r.json()["data"]["job_id"])
    assert job["status"] == "succeeded" and job["log_tail"][-1] == "line 2"


def test_collect_same_item_twice_409(client):
    r1 = client.post("/api/collect/Test Ore")
    r2 = client.post("/api/collect/Test Ore")
    assert r1.status_code == 202 and r2.status_code == 409
    _wait(client, r1.json()["data"]["job_id"])


def test_collect_unknown_item_404(client):
    assert client.post("/api/collect/Ghost").status_code == 404


def test_scheduler_status_and_tasks(client, project_root):
    (project_root / "src" / "scheduler").mkdir()
    (project_root / "src" / "scheduler" / "task_config.py").write_text(
        "TASKS=[{'name':'T','script_path':'src/api/Test_Ore_API.py','schedule_type':'daily',"
        "'schedule_value':'23:00','enabled':True,'description':'d'}]\n")
    d = client.get("/api/scheduler").json()["data"]
    assert d["running"] is False and d["tasks"][0]["item"] == "Test Ore"
    r = client.post("/api/scheduler/run", json={"script_path": "src/api/Test_Ore_API.py"})
    assert r.status_code == 202
    assert _wait(client, r.json()["data"]["job_id"])["status"] == "succeeded"
    assert client.post("/api/scheduler/run", json={"script_path": "nope.py"}).status_code == 404
    assert client.post("/api/scheduler/run", json={"script_path": "../../etc/passwd"}).status_code == 422


def test_scheduler_start_stop(client, project_root):
    (project_root / "src" / "scheduler").mkdir(exist_ok=True)
    (project_root / "src" / "scheduler" / "start_scheduler.py").write_text("import time\nwhile True: time.sleep(0.1)\n")
    assert client.post("/api/scheduler/start").json()["data"]["running"] is True
    assert client.get("/api/scheduler").json()["data"]["running"] is True
    assert client.post("/api/scheduler/stop").json()["data"]["running"] is False


def test_items_config_roundtrip(client, project_root):
    cfg = client.get("/api/config/items").json()["data"]
    cfg["items"][0]["description"] = "changed"
    assert client.put("/api/config/items", json=cfg).status_code == 200
    assert json.loads((project_root / "data" / "items.json").read_text())["items"][0]["description"] == "changed"
    bad = {**cfg, "items": cfg["items"] + [cfg["items"][0]]}
    assert client.put("/api/config/items", json=bad).status_code == 422
