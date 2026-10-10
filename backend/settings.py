from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Environment variables (or .env in the project root) override defaults."""

    model_config = SettingsConfigDict(env_file=str(DEFAULT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    # optional Postgres; leave DB_USER empty to use CSV files in data/
    db_host: str = "localhost"
    db_database: str = "darkerdb"
    db_user: str = ""
    db_password: str = ""
    db_port: int = 5432
    db_connect_timeout: int = 3

    darker_market_api_key: str = ""

    project_root: Path = Field(default=DEFAULT_ROOT)
    host: str = "127.0.0.1"  # single-user local app; set HOST=0.0.0.0 to expose on the LAN
    port: int = 5001
    # env DARKER_DATA_DIR. Not DATA_DIR: older .env files use that for an unrelated export folder.
    darker_data_dir: Path | None = None
    runs_dir: Path | None = None
    seasons_path: Path | None = None
    items_config_path: Path | None = None
    catalog_path: Path | None = None
    env_path: Path | None = None
    app_settings_path: Path | None = None
    sqlite_path: Path | None = None
    frontend_dist: Path | None = None
    python_executable: str = sys.executable

    def model_post_init(self, __context) -> None:
        root = self.project_root
        self.darker_data_dir = self.darker_data_dir or root / "data"
        self.runs_dir = self.runs_dir or self.data_dir / "runs"
        self.seasons_path = self.seasons_path or self.data_dir / "seasons.json"
        self.items_config_path = self.items_config_path or self.data_dir / "items.json"
        self.catalog_path = self.catalog_path or self.data_dir / "catalog.json"
        self.env_path = self.env_path or root / ".env"
        self.app_settings_path = self.app_settings_path or self.darker_data_dir / "settings.json"
        self.sqlite_path = self.sqlite_path or self.darker_data_dir / "market.db"
        self.frontend_dist = self.frontend_dist or root / "frontend" / "dist"

    @property
    def data_dir(self) -> Path:
        return self.darker_data_dir

    @property
    def postgres_configured(self) -> bool:
        return bool(self.db_user)

    @property
    def default_items_path(self) -> Path:
        return self.project_root / "items_config.json"

    @property
    def csv_dirs(self) -> list[Path]:
        """Where collected CSVs are looked up: data/ first, then the project root (pre-data/ layout)."""
        return [self.data_dir, self.project_root]

    def bootstrap(self) -> None:
        """Create data/ and seed user state on first start. Never overwrites existing files."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.items_config_path.exists():
            if self.default_items_path.exists():
                shutil.copyfile(self.default_items_path, self.items_config_path)
            else:
                self.items_config_path.write_text(json.dumps({"items": [], "categories": {}}, indent=2))
        if not self.seasons_path.exists():
            legacy = self.project_root / "seasons.json"
            if legacy.exists():
                shutil.copyfile(legacy, self.seasons_path)
            else:
                self.seasons_path.write_text(json.dumps({"seasons": []}))
        legacy_runs = self.project_root / "runs"
        if legacy_runs.is_dir() and legacy_runs != self.runs_dir:
            for f in legacy_runs.glob("*.json"):
                target = self.runs_dir / f.name
                if not target.exists():
                    shutil.copyfile(f, target)
