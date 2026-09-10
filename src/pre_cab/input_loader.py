"""Shared local JSON export loader used by profiling and benchmark tools."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_cr_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CR records from a bare list or common ServiceNow/export wrappers.

    Accepted roots are a list of records or an object containing records under ``result``,
    ``records``, ``changes``, or ``data``. A single mapping representing one CR is also accepted.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "records", "changes", "data"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        if any(key in payload for key in ("Number", "number", "sys_id", "short_description")):
            return [payload]
    raise ValueError("Input JSON must contain CR records as a list or a supported wrapper object")
