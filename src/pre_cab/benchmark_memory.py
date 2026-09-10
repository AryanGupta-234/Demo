"""Build benchmark-safe memory from training/reference records without test leakage."""
from __future__ import annotations

from typing import Any, Iterable

from .cab_outcomes import normalize_cab_recommendation
from .memory import MemoryKind, MemoryRecord, UnifiedMemory


def build_reference_memory(records: Iterable[dict[str, Any]], memory: UnifiedMemory) -> int:
    """Index only supplied reference records; callers should pass train/reference splits, not test rows."""
    count = 0
    for row in records:
        source_id = str(row.get("Number") or row.get("Effective number") or "unknown")
        outcome = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
        text = " ".join(str(row.get(k) or "") for k in (
            "Short description", "Description", "Justification", "Category",
            "Sub Category", "Implementation plan", "Test plan", "Backout plan",
        ))
        metadata = {
            "source_id": source_id,
            "type": row.get("Type"),
            "category": row.get("Category"),
            "sub_category": row.get("Sub Category"),
            "cab_outcome": outcome.value if outcome else None,
        }
        kind = MemoryKind.SIMILARITY if outcome is None else MemoryKind.CAB_HISTORY
        memory.add(MemoryRecord(key=source_id, kind=kind, text=text, metadata=metadata))
        count += 1
    return count
