from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_host: str = "localhost"
    db_database: str = "darkerdb"
    db_user: str = ""
    db_password: str = ""
    db_port: int = 5432
    db_connect_timeout: int = 3
    project_root: Path = Field(default=DEFAULT_ROOT)
    port: int = 5001
    host: str = "127.0.0.1"  # single-user local app; set HOST=0.0.0.0 to expose on the LAN
    runs_dir: Path | None = None
    seasons_path: Path | None = None
    items_config_path: Path | None = None
    frontend_dist: Path | None = None
    python_executable: str = sys.executable

    def model_post_init(self, __context) -> None:
        root = self.project_root
        self.runs_dir = self.runs_dir or root / "runs"
        self.seasons_path = self.seasons_path or root / "seasons.json"
        self.items_config_path = self.items_config_path or root / "items_config.json"
        self.frontend_dist = self.frontend_dist or root / "frontend" / "dist"

    @property
    def postgres_configured(self) -> bool:
        return bool(self.db_user)
