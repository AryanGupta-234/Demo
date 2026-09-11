"""Minimal dependency-free .env loader for local development."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: str | Path | None = None) -> Path | None:
    """Load a local .env file without overwriting existing environment variables."""
    base = Path(start or Path.cwd()).resolve()
    candidates = [base / ".env", *[parent / ".env" for parent in base.parents]]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
        return path
    return None
