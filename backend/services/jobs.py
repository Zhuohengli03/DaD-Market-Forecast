from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TERMINAL = ("succeeded", "failed")
_SENTINEL = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None
    result_ref: str | None = None
    meta: dict = field(default_factory=dict)
    progress: dict | None = None
    cost: dict = field(default_factory=lambda: {"estimate": None, "used": 0})   # DarkerDB credits
    _log: deque = field(default_factory=lambda: deque(maxlen=2000), repr=False)

    def log_tail(self, n: int | None = 200) -> list[str]:
        lines = list(self._log)
        return lines if n is None else lines[-n:]

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                "created_at": self.created_at, "finished_at": self.finished_at, "error": self.error,
                "result_ref": self.result_ref, "meta": self.meta, "progress": self.progress,
                "cost": dict(self.cost)}
        # raw log lines stay server-side (troubleshooting); the UI shows structured progress only


class JobManager:
    def __init__(self, max_jobs: int = 200):
        self._max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._subs: dict[str, list[queue.Queue]] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ----- registry -----
    def create(self, kind: str, label: str, meta: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
            finished = sorted((j for j in self._jobs.values() if j.status in TERMINAL), key=lambda j: j.created_at)
            for old in finished[: max(0, len(self._jobs) - self._max_jobs)]:
                del self._jobs[old.id]
                self._subs.pop(old.id, None)
        return job

    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def active(self) -> list[Job]:
        return [j for j in self.list() if j.status in ("queued", "running")]

    def find_active(self, kind: str, **meta) -> Job | None:
        for j in self.active():
            if j.kind == kind and all(j.meta.get(k) == v for k, v in meta.items()):
                return j
        return None

    # ----- events -----
    def _publish(self, job: Job, event: str, data: str) -> None:
        for q in list(self._subs.get(job.id, [])):
            q.put((event, data))

    def append_log(self, job: Job, line: str) -> None:
        with self._lock:
            job._log.append(line)

    def set_estimate(self, job: Job, requests: int | None) -> None:
        with self._lock:
            job.cost = {**job.cost, "estimate": requests}

    def add_credits(self, job: Job, n: int | None) -> None:
        if n:
            with self._lock:
                job.cost = {**job.cost, "used": job.cost["used"] + int(n)}

    def progress(self, job: Job, stage: str, done: int | None = None, total: int | None = None,
                 detail: dict | None = None, **kw) -> None:
        """Structured progress for the UI: a stage code, optional done/total counts and details."""
        with self._lock:
            job.progress = {"stage": stage, "done": done, "total": total, "detail": {**(detail or {}), **kw}}
            self._publish(job, "progress", json.dumps(job.progress, ensure_ascii=False))

    def _set_status(self, job: Job, status: str, error: str | None = None, result_ref: str | None = None) -> None:
        with self._lock:
            job.status = status
            if error is not None:
                job.error = error
            if result_ref is not None:
                job.result_ref = result_ref
            if status in TERMINAL:
                job.finished_at = _now()
            self._publish(job, "status", status)
            if status in TERMINAL:
                for q in list(self._subs.get(job.id, [])):
                    q.put(_SENTINEL)

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            job = self._jobs[job_id]
            self._subs.setdefault(job_id, []).append(q)
            if job.progress is not None:  # late subscribers start from the current progress
                q.put(("progress", json.dumps(job.progress, ensure_ascii=False)))
            q.put(("status", job.status))
            if job.status in TERMINAL:
                q.put(_SENTINEL)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs.get(job_id, []):
                self._subs[job_id].remove(q)

    async def events(self, job_id: str):
        q = self.subscribe(job_id)
        try:
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.2)
                    continue
                if item is _SENTINEL:
                    return
                event, data = item
                yield {"event": event, "data": data}
        finally:
            self.unsubscribe(job_id, q)

    # ----- execution -----
    def run_in_thread(self, job: Job, fn: Callable[[Job], str | None]) -> None:
        def target():
            self._set_status(job, "running")
            try:
                ref = fn(job)
                self._set_status(job, "succeeded", result_ref=ref)
            except Exception as exc:
                self.append_log(job, f"error: {type(exc).__name__}: {exc}")
                self._set_status(job, "failed", error=f"{type(exc).__name__}: {exc}")
        threading.Thread(target=target, daemon=True, name=f"job-{job.id}").start()

    def run_process(self, job: Job, argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
        def target():
            self._set_status(job, "running")
            try:
                proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                                        env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"})
                with self._lock:
                    self._procs[job.id] = proc
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip("\n")
                    if line.startswith("@@"):  # structured progress from the child process
                        try:
                            obj = json.loads(line[2:])
                            credits = obj.pop("credits", None)
                            if credits is not None:  # cumulative for this one process
                                with self._lock:
                                    job.cost = {**job.cost, "used": int(credits)}
                            self.progress(job, obj.pop("stage", "running"), obj.pop("done", None),
                                          obj.pop("total", None), detail=obj)
                            continue
                        except (ValueError, AttributeError):
                            pass
                    self.append_log(job, line)
                code = proc.wait()
                if code == 0:
                    self._set_status(job, "succeeded")
                else:
                    self._set_status(job, "failed", error=f"exit code {code}")
            except Exception as exc:
                self._set_status(job, "failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._procs.pop(job.id, None)
        threading.Thread(target=target, daemon=True, name=f"job-{job.id}").start()

    def shutdown(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
