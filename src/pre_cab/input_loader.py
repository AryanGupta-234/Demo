"""Shared local JSON export loader used by profiling and benchmark tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _find_records(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        # Prefer common ServiceNow/export envelopes first.
        for key in ("result", "records", "changes", "data", "items", "results"):
            if key in payload:
                found = _find_records(payload[key])
                if found is not None:
                    return found
        # Some exports wrap records one or more levels deeper under arbitrary keys.
        for value in payload.values():
            found = _find_records(value)
            if found is not None:
                return found
    return None


def load_cr_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CR records from bare lists or nested ServiceNow/export envelopes."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = _find_records(payload)
    if records is None:
        raise ValueError("Input JSON does not contain CR records as a list or nested export envelope")
    return records


def first_value(record: dict[str, Any], *names: str) -> Any:
    """Case-insensitive field lookup for common ServiceNow/export naming variants."""
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.strip().lower())
        if value is not None:
            return value
    return None


def record_type(record: dict[str, Any]) -> str:
    value = first_value(record, "Type", "change type", "change_type", "type")
    return str(value or "").strip().lower()


def source_id(record: dict[str, Any]) -> str:
    value = first_value(record, "Number", "Effective number", "number", "effective_number", "Change Number", "change_number")
    return str(value or "unknown")
