"""Prepare historical Normal CRs for leakage-safe benchmark replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .schemas import Decision


@dataclass(frozen=True)
class BenchmarkExample:
    input_record: dict[str, Any]
    actual: Decision | None
    source_id: str


def normalize_outcome(record: dict[str, Any]) -> Decision | None:
    """Map explicit historical outcome text to the validator's three-way benchmark label.

    Unknown or ambiguous CAB text remains unscored rather than being guessed.
    """
    raw = str(record.get("CAB Outcome") or record.get("CAB recommendation") or "").strip().lower()
    if not raw:
        return None
    if any(term in raw for term in ("not ready", "rejected", "reject", "hold", "held")):
        return Decision.NOT_READY
    if any(term in raw for term in ("conditional", "condition", "clarification", "proceed with condition")):
        return Decision.CONDITIONAL
    if any(term in raw for term in ("approved", "approve", "proceed", "ok")):
        return Decision.PASS
    return None


def prepare_normal_benchmark(records: Iterable[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for original in records:
        if str(original.get("Type") or "").strip().lower() != "normal":
            continue
        actual = normalize_outcome(original)
        hidden = dict(original)
        # Do not let the target answer leak into model-visible input.
        for key in ("CAB Outcome", "CAB recommendation", "CAB Recommendation", "historical_prediction_label"):
            hidden.pop(key, None)
        source_id = str(original.get("Number") or original.get("Effective number") or "unknown")
        examples.append(BenchmarkExample(hidden, actual, source_id))
    return examples
