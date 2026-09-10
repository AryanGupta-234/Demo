"""Prepare historical Normal CRs for leakage-safe benchmark replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .benchmark_leakage import strip_post_decision_fields
from .schemas import Decision


@dataclass(frozen=True)
class BenchmarkExample:
    input_record: dict[str, Any]
    actual: Decision | None
    source_id: str


def normalize_outcome(record: dict[str, Any]) -> Decision | None:
    """Map explicit historical CAB text to a three-way benchmark label.

    Ambiguous operational notes remain unscored instead of being guessed.
    """
    raw = str(record.get("CAB Outcome") or record.get("CAB recommendation") or "").strip().lower()
    if not raw:
        return None
    if any(term in raw for term in (
        "cancellation requested", "request for cancellation", "cancelled as",
        "cancelled", "cancel", "rejected", "reject", "insufficient information", "hold", "held",
    )):
        return Decision.NOT_READY
    if any(term in raw for term in (
        "conditionally approved", "conditional", "condition", "test results reqd",
        "schedule needs to be updated",
    )):
        return Decision.CONDITIONAL
    if raw in {"approved", "approve", "approvd", "approved."} or raw.startswith("approved "):
        return Decision.PASS
    return None


def prepare_normal_benchmark(records: Iterable[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for original in records:
        if str(original.get("Type") or "").strip().lower() != "normal":
            continue
        actual = normalize_outcome(original)
        hidden = strip_post_decision_fields(dict(original))
        for key in ("CAB Outcome", "CAB recommendation", "CAB Recommendation", "historical_prediction_label"):
            hidden.pop(key, None)
        source_id = str(original.get("Number") or original.get("Effective number") or "unknown")
        examples.append(BenchmarkExample(hidden, actual, source_id))
    return examples
