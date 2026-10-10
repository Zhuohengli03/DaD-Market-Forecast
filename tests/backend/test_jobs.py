import time

from backend.services.jobs import JobManager


def _wait(client, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()["data"]
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_thread_job_success_and_logs():
    jm = JobManager()
    job = jm.create("test", "t")

    def work(j):
        jm.append_log(j, "hello")
        return "ref-1"
    jm.run_in_thread(job, work)
    for _ in range(100):
        if job.status == "succeeded":
            break
        time.sleep(0.02)
    assert job.status == "succeeded" and job.result_ref == "ref-1" and job.log_tail() == ["hello"]
    jm.shutdown()


def test_thread_job_failure_records_error():
    jm = JobManager()
    job = jm.create("test", "t")

    def work(j):
        raise ValueError("nope")
    jm.run_in_thread(job, work)
    for _ in range(100):
        if job.status == "failed":
            break
        time.sleep(0.02)
    assert job.status == "failed" and "nope" in job.error
    jm.shutdown()


def test_process_job_streams_stdout(client, project_root):
    jm = client.app.state.jobs
    job = jm.create("collect", "test", meta={"item": "Test Ore"})
    jm.run_process(job, [client.app.state.settings.python_executable, str(project_root / "src/api/Test_Ore_API.py")],
                   project_root)
    done = _wait(client, job.id)
    assert done["status"] == "succeeded"
    assert jm.get(job.id).log_tail() == ["line 0", "line 1", "line 2"]          # server-side only
    assert "log_tail" not in client.get(f"/api/jobs/{job.id}").json()["data"]


def test_sse_events_end_on_completion(client, project_root):
    jm = client.app.state.jobs
    job = jm.create("collect", "test", meta={"item": "Test Ore"})
    jm.run_process(job, [client.app.state.settings.python_executable, str(project_root / "src/api/Test_Ore_API.py")],
                   project_root)
    events = []
    with client.stream("GET", f"/api/jobs/{job.id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert "log" not in events and events[-1] == "status"


def test_jobs_list_and_404(client):
    assert client.get("/api/jobs").json()["ok"] is True
    assert client.get("/api/jobs/zzz").status_code == 404
