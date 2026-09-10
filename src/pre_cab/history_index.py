"""Index historical Normal CRs into any UnifiedMemory implementation."""
from __future__ import annotations

from typing import Any, Iterable

from .benchmark_leakage import strip_post_decision_fields
from .cab_outcomes import normalize_cab_recommendation
from .memory import MemoryKind, MemoryRecord, UnifiedMemory
from .sanitization import sanitize_value

_HISTORY_FIELDS = (
    "Short description",
    "Description",
    "Justification",
    "Category",
    "Sub Category",
    "Change Class",
    "Implementation plan",
    "Backout plan",
    "Test plan",
    "Configuration item",
    "Risk",
    "Risk and impact analysis",
)


def history_record(row: dict[str, Any]) -> MemoryRecord | None:
    if str(row.get("Type") or "").strip().lower() != "normal":
        return None
    number = str(row.get("Number") or row.get("Effective number") or "").strip()
    if not number:
        return None
    clean = sanitize_value(strip_post_decision_fields(row))
    parts: list[str] = []
    for field in _HISTORY_FIELDS:
        value = str(clean.get(field) or "").strip()
        if value:
            parts.append(f"{field}: {value}")
    if not parts:
        return None
    outcome = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
    return MemoryRecord(
        memory_id=f"history:{number}",
        kind=MemoryKind.CAB_HISTORY,
        text="\n".join(parts),
        metadata={
            "cr_number": number,
            "historical_outcome": outcome.value if outcome else None,
            "category": row.get("Category"),
            "sub_category": row.get("Sub Category"),
            "change_class": row.get("Change Class"),
            "source_record": clean,
        },
    )


def index_history(memory: UnifiedMemory, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    for row in records:
        record = history_record(row)
        if record is None:
            continue
        memory.remember(record)
        count += 1
    return count
