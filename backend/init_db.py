"""Prepare user state and the built-in database. Called by start.sh / start.bat before the server starts.

    python -m backend.init_db [--project-root PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.api.sqlite_db import connect
from .data.sqlite_repo import import_csv_once
from .settings import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    args = p.parse_args(argv)
    s = Settings(project_root=Path(args.project_root)) if args.project_root else Settings()
    s.bootstrap()
    if s.postgres_configured:
        print(f"数据库：Postgres {s.db_host}/{s.db_database}（未连接时自动改用内置数据库）")
    connect(s.sqlite_path).close()
    for name, n in import_csv_once(s).items():
        print(f"已把 {name} 的 {n} 条旧 CSV 数据导入内置数据库")
    print(f"内置数据库：{s.sqlite_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
