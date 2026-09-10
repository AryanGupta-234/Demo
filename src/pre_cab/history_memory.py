"""Build retrieval memory from reference/training CRs without exposing holdout outcomes."""
from __future__ import annotations

from typing import Any, Iterable

from .benchmark_leakage import strip_post_decision_fields
from .cab_outcomes import normalize_cab_recommendation
from .memory import InMemoryUnifiedMemory, MemoryKind, MemoryRecord
from .sanitization import sanitize_value


_TEXT_FIELDS = (
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
)


def _history_text(row: dict[str, Any]) -> str:
    clean = strip_post_decision_fields(row)
    parts = []
    for field in _TEXT_FIELDS:
        value = str(clean.get(field) or "").strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def build_history_memory(
    reference_rows: Iterable[dict[str, Any]],
    *,
    embedding_provider: Any | None = None,
) -> InMemoryUnifiedMemory:
    """Create memory from non-holdout Normal CRs.

    Reference outcomes may be retained as explicit historical metadata, while ``source_record`` is
    stripped of post-decision fields. Holdout/test rows must never be passed to this function.
    """
    memory = InMemoryUnifiedMemory(embedding_provider=embedding_provider)
    for row in reference_rows:
        if str(row.get("Type") or "").strip().lower() != "normal":
            continue
        number = str(row.get("Number") or row.get("Effective number") or "unknown").strip()
        outcome = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
        clean_source = sanitize_value(strip_post_decision_fields(row))
        text = _history_text(row)
        if not text:
            continue
        memory.remember(
            MemoryRecord(
                memory_id=f"history:{number}",
                kind=MemoryKind.CAB_HISTORY,
                text=text,
                metadata={
                    "cr_number": number,
                    "historical_outcome": outcome.value if outcome else None,
                    "category": row.get("Category"),
                    "sub_category": row.get("Sub Category"),
                    "change_class": row.get("Change Class"),
                    "source_record": clean_source,
                },
            )
        )
    return memory
