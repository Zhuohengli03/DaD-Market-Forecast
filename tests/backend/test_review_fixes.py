import os
import subprocess
import sys
import threading
import time

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.jobs import JobManager
from backend.settings import Settings


def _client_with_dist(project_root):
    dist = project_root / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=\"root\"></div>")
    (project_root / ".env").write_text("DB_PASSWORD=supersecret\n")
    return TestClient(create_app(Settings(project_root=project_root, db_user="")), headers={"X-Darker-Market": "1"})


def test_spa_fallback_does_not_escape_dist(project_root):
    with _client_with_dist(project_root) as c:
        for path in ("/%2e%2e/%2e%2e/.env", "/..%2f..%2f.env"):
            r = c.get(path)
            assert "supersecret" not in r.text
        assert c.get("/forecast").text.startswith("<div id=\"root\">")


def test_default_bind_is_localhost():
    assert Settings().host == "127.0.0.1"


def test_season_empty_start_is_422(client):
    r = client.post("/api/seasons", json={"name": "X", "start": "", "end": None})
    assert r.status_code == 422


def test_process_output_streams_before_exit(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('first')\ntime.sleep(1.5)\nprint('second')\n")  # no flush
    jm = JobManager()
    job = jm.create("task", "slow")
    jm.run_process(job, [sys.executable, str(script)], tmp_path)
    t0 = time.time()
    while time.time() - t0 < 1.2 and "first" not in job.log_tail():
        time.sleep(0.05)
    assert "first" in job.log_tail(), "stdout was block-buffered"
    while job.status not in ("succeeded", "failed"):
        time.sleep(0.05)
    jm.shutdown()


def test_scheduler_terminated_on_shutdown(project_root):
    (project_root / "src" / "scheduler").mkdir(exist_ok=True)
    (project_root / "src" / "scheduler" / "start_scheduler.py").write_text("import time\nwhile True: time.sleep(0.1)\n")
    app = create_app(Settings(project_root=project_root, db_user=""))
    with TestClient(app, headers={"X-Darker-Market": "1"}) as c:
        c.post("/api/scheduler/start")
        proc = app.state.scheduler._proc
        assert proc.poll() is None
    proc.wait(timeout=5)
    assert proc.poll() is not None


def test_status_caches_item_stats(project_root):
    calls = []
    app = create_app(Settings(project_root=project_root, db_user=""))
    with TestClient(app, headers={"X-Darker-Market": "1"}) as c:
        real = app.state.repo.list_items
        app.state.repo.list_items = lambda: calls.append(1) or real()
        for _ in range(5):
            assert c.get("/api/status").json()["data"]["items"]
    assert len(calls) <= 1


def test_job_list_safe_under_concurrent_create():
    jm = JobManager()
    errors = []
    stop = threading.Event()

    def creator():
        while not stop.is_set():
            jm.create("t", "x")

    th = threading.Thread(target=creator)
    th.start()
    try:
        for _ in range(3000):
            try:
                jm.list(); jm.active()
            except RuntimeError as e:
                errors.append(e)
                break
    finally:
        stop.set(); th.join()
    assert not errors


def test_internal_errors_do_not_leak_details(client):
    def boom():
        raise RuntimeError("password=hunter2 in SQL")
    client.app.state.item_stats._fn = boom
    client.app.state.item_stats.invalidate()
    r = TestClient(client.app, raise_server_exceptions=False).get("/api/status")
    assert r.status_code == 500 and "hunter2" not in r.text and "ref" in r.json()["error"]


def test_late_subscriber_gets_current_progress_and_old_jobs_are_evicted():
    jm = JobManager(max_jobs=3)
    job = jm.create("t", "x")
    for i in range(500):
        jm.progress(job, "collect", i + 1, 500)
    jm._set_status(job, "succeeded")
    q = jm.subscribe(job.id)
    events = []
    while True:
        item = q.get_nowait()
        if item is None:
            break
        events.append(item[0])
    assert events == ["progress", "status"]
    for _ in range(5):
        j = jm.create("t", "y"); jm._set_status(j, "succeeded")
    assert len(jm.list()) == 3
