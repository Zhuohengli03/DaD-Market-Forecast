from __future__ import annotations

import subprocess
from pathlib import Path

from ..settings import Settings
from .jobs import Job, JobManager


class SchedulerController:
    def __init__(self, settings: Settings, jobs: JobManager):
        self._settings = settings
        self._jobs = jobs
        self._proc: subprocess.Popen | None = None
        self._tasks_cache: tuple[tuple, list[dict]] | None = None

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
                                      env={**__import__("os").environ, "DARKER_ITEMS_CONFIG": str(self._settings.items_config_path),
                                           "DARKER_DATA_DIR": str(self._settings.data_dir),
                                           "DARKER_SETTINGS": str(self._settings.app_settings_path)},
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def restart_if_running(self) -> None:
        """The scheduler reads tracked items at start; restart it so item or time changes take effect."""
        self._tasks_cache = None
        if self.running():
            self.stop()
            self.start()

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
        """One daily collection task per enabled tracked item (same rule as src/scheduler/task_config.py)."""
        path = self._settings.items_config_path
        sp = self._settings.app_settings_path
        mtime = (path.stat().st_mtime if path.exists() else 0.0, sp.stat().st_mtime if sp.exists() else 0.0)
        if self._tasks_cache and self._tasks_cache[0] == mtime:
            return self._tasks_cache[1]
        from src.scheduler.task_config import tasks_from_items
        out = [{"name": t["name"], "item": t["item"], "script_path": t["script_path"],
                "schedule_type": t["schedule_type"], "schedule_value": str(t["schedule_value"]),
                "enabled": t["enabled"], "description": t["description"]}
               for t in tasks_from_items(path, self._settings.project_root, settings_path=sp)]
        self._tasks_cache = (mtime, out)
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
