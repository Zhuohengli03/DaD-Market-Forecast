import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "scheduler"))


def test_seconds_schedule_and_no_pile_up(tmp_path, monkeypatch):
    import core_scheduler
    import schedule
    schedule.clear()
    sched = core_scheduler.SmartScheduler.__new__(core_scheduler.SmartScheduler)
    sched.running_tasks = {}
    sched._queued = set()
    sched._queued_lock = threading.Lock()
    script = tmp_path / "s.py"
    script.write_text("import time\ntime.sleep(0.6)\n")
    sched.add_task_from_config({"name": "Collect X", "script_path": str(script), "args": [], "schedule_type": "seconds",
                                "schedule_value": 30, "enabled": True})
    job = schedule.get_jobs()[0]
    assert job.interval == 30 and job.unit == "seconds"
    first = sched.run_script_async(str(script), "Collect X")
    second = sched.run_script_async(str(script), "Collect X")      # previous run still going: skipped
    assert first is not None and second is None
    first.join(timeout=5)
    assert sched.run_script_async(str(script), "Collect X") is not None
    schedule.clear()
