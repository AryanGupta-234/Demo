"""Prepare historical Normal CRs for leakage-safe benchmark replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .benchmark_leakage import strip_post_decision_fields
from .input_loader import first_value, record_type, source_id
from .schemas import Decision


@dataclass(frozen=True)
class BenchmarkExample:
    input_record: dict[str, Any]
    actual: Decision | None
    source_id: str


def _field_text(record: dict[str, Any], *names: str) -> str:
    value = first_value(record, *names)
    return str(value or "").strip().lower()


def _is_normal(record: dict[str, Any]) -> bool:
    value = record_type(record)
    if value in {"normal", "normal change", "normal-change"}:
        return True
    # Handle exports that spell the field differently, e.g. change_type_display.
    for key, raw in record.items():
        key_text = str(key).strip().lower().replace("_", " ")
        if "type" in key_text or "change class" in key_text or "change category" in key_text:
            normalized = str(raw or "").strip().lower()
            if normalized in {"normal", "normal change", "normal-change"}:
                return True
    return False


def normalize_outcome(record: dict[str, Any]) -> Decision | None:
    """Map explicit historical CAB text to a three-way benchmark label.

    Ambiguous operational notes remain unscored instead of being guessed.
    """
    raw = _field_text(record, "CAB Outcome", "CAB recommendation", "CAB Recommendation", "cab_outcome", "cab_recommendation")
    if not raw:
        return None
    if any(term in raw for term in (
        "cancellation requested", "request for cancellation", "cancelled as",
        "cancelled", "cancel", "rejected", "reject", "insufficient information", "hold", "held",
    )):
        return Decision.NOT_READY
    if any(term in raw for term in (
        "conditionally approved", "conditional", "condition", "test results reqd",
        "schedule needs to be updated", "approval pending", "customer approval pending",
    )):
        return Decision.CONDITIONAL
    if raw in {"approved", "approve", "approvd", "approved."} or raw.startswith("approved "):
        return Decision.PASS
    return None


def prepare_normal_benchmark(records: Iterable[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for original in records:
        if not _is_normal(original):
            continue
        actual = normalize_outcome(original)
        hidden = strip_post_decision_fields(dict(original))
        for key in (
            "CAB Outcome", "CAB recommendation", "CAB Recommendation",
            "cab_outcome", "cab_recommendation", "historical_prediction_label",
        ):
            hidden.pop(key, None)
        examples.append(BenchmarkExample(hidden, actual, source_id(original)))
    return examples
