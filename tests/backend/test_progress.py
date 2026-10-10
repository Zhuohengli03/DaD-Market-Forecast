import json
import sys
import time

from backend.services.jobs import JobManager


def _wait(job, timeout=10):
    t0 = time.time()
    while job.status not in ("succeeded", "failed") and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_progress_is_stored_and_published():
    jm = JobManager()
    job = jm.create("t", "x")
    q = jm.subscribe(job.id)
    jm.progress(job, "backtest", 2, 7, model="ridge")
    assert job.to_dict()["progress"] == {"stage": "backtest", "done": 2, "total": 7, "detail": {"model": "ridge"}}
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert ("progress", json.dumps(job.progress, ensure_ascii=False)) in events


def test_process_progress_lines_become_progress_not_log(tmp_path):
    script = tmp_path / "p.py"
    script.write_text('print("plain log line")\nprint(\'@@{"stage": "collect", "done": 1, "total": 3, "new": 50}\')\n'
                      'print(\'@@{"stage": "done", "new": 50}\')\n')
    jm = JobManager()
    job = jm.create("collect", "c")
    jm.run_process(job, [sys.executable, str(script)], tmp_path)
    _wait(job)
    assert job.status == "succeeded"
    assert job.progress == {"stage": "done", "done": None, "total": None, "detail": {"new": 50}}
    assert all(not line.startswith("@@") for line in job.log_tail())


def test_thread_job_summary_from_progress():
    jm = JobManager()
    job = jm.create("t", "x")

    def work(j):
        jm.progress(j, "done", detail={"items": 3})
        return "ref"
    jm.run_in_thread(job, work)
    _wait(job)
    assert job.to_dict()["progress"]["detail"] == {"items": 3}


def test_job_dict_does_not_expose_logs_to_ui():
    jm = JobManager()
    job = jm.create("t", "x")
    jm.append_log(job, "internal detail")
    d = job.to_dict()
    assert "log_tail" not in d and d["progress"] is None
