from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    """Write to a temp file in the same directory, then rename over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def upsert_env(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a dotenv file, keeping every other line as is. The file is made owner-only."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    atomic_write(path, "\n".join(out) + "\n", mode=0o600)
