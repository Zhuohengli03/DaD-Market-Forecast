from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from ..settings import Settings
from .jobs import Job, JobManager


class SchedulerController:
    def __init__(self, settings: Settings, jobs: JobManager):
        self._settings = settings
        self._jobs = jobs
        self._proc: subprocess.Popen | None = None

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        if self.running():
            return True
        script = self._settings.project_root / "src" / "scheduler" / "start_scheduler.py"
        if not script.exists():
            raise FileNotFoundError(f"missing {script}")
        self._proc = subprocess.Popen([self._settings.python_executable, str(script), "--start"],
                                      cwd=str(self._settings.project_root),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def stop(self) -> bool:
        if self.running():
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        return False

    def tasks(self) -> list[dict]:
        path = self._settings.project_root / "src" / "scheduler" / "task_config.py"
        if not path.exists():
            return []
        spec = importlib.util.spec_from_file_location("task_config_dyn", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        root = str(self._settings.project_root)
        out = []
        for t in getattr(mod, "TASKS", []):
            sp = Path(t["script_path"])
            rel = str(sp.relative_to(root)) if sp.is_absolute() and str(sp).startswith(root) else str(sp)
            out.append({"name": t["name"], "script_path": rel, "schedule_type": t["schedule_type"],
                        "schedule_value": str(t["schedule_value"]), "enabled": t.get("enabled", True),
                        "description": t.get("description", "")})
        return out

    def resolve_script(self, script_path: str) -> Path:
        root = self._settings.project_root.resolve()
        p = Path(script_path)
        p = p.resolve() if p.is_absolute() else (root / p).resolve()
        if root not in p.parents:
            raise ValueError("script_path must be inside the project")
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    def run_task(self, script_path: str) -> Job:
        p = self.resolve_script(script_path)
        job = self._jobs.create("task", f"run {p.name}", meta={"script": str(p)})
        self._jobs.run_process(job, [self._settings.python_executable, str(p)], self._settings.project_root)
        return job
