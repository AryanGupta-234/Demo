"""Prepare historical Normal CRs for leakage-safe benchmark replay."""
from __future__ import annotations

from collections import Counter
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


def _normalized_key(key: object) -> str:
    return "".join(ch for ch in str(key).strip().lower() if ch.isalnum())


def _normalize_type(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _is_normal(record: dict[str, Any]) -> bool:
    candidates = [record_type(record)]
    for key, raw in record.items():
        key_text = _normalized_key(key)
        if "changetype" in key_text or "changeclass" in key_text or "changeclassification" in key_text:
            candidates.append(str(raw or ""))
    for candidate in candidates:
        normalized = _normalize_type(candidate)
        if normalized in {"normal", "normal change", "normalchange"}:
            return True
        if "normal change" in normalized:
            return True
    return False


def _outcome_text(record: dict[str, Any]) -> str:
    direct = _field_text(
        record,
        "CAB Outcome", "CAB recommendation", "CAB Recommendation", "cab_outcome", "cab_recommendation",
        "CAB Decision", "CAB Status", "Final CAB Decision", "Final CAB Outcome",
        "CAB Decision/Recommendation", "CAB Comments", "CAB Comment",
    )
    if direct:
        return direct
    for key, raw in record.items():
        key_text = _normalized_key(key)
        if "cab" in key_text and any(token in key_text for token in ("outcome", "recommendation", "decision", "status", "result", "comment")):
            text = str(raw or "").strip().lower()
            if text:
                return text
    return ""


def normalize_outcome(record: dict[str, Any]) -> Decision | None:
    """Map explicit historical CAB text to a three-way benchmark label.

    Ambiguous operational notes remain unscored instead of being guessed.
    """
    raw = _outcome_text(record)
    if not raw:
        return None
    if any(term in raw for term in (
        "cancellation requested", "request for cancellation", "cancelled as",
        "cancelled", "canceled", "cancel", "rejected", "reject", "insufficient information",
        "not ready", "not approved", "hold", "held", "do not approve", "don't approve",
    )):
        return Decision.NOT_READY
    if any(term in raw for term in (
        "conditionally approved", "conditional approval", "conditional", "condition",
        "test results reqd", "test results required", "schedule needs to be updated",
        "approval pending", "customer approval pending", "pending approval", "approve with conditions",
    )):
        return Decision.CONDITIONAL
    if any(term in raw for term in (
        "approved", "approve", "approvd", "approved for implementation", "go ahead", "okay to proceed",
    )) and not any(term in raw for term in ("not approved", "not approve", "pending approval")):
        return Decision.PASS
    return None


def prepare_normal_benchmark(records: Iterable[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for original in records:
        if not _is_normal(original):
            continue
        actual = normalize_outcome(original)
        hidden = strip_post_decision_fields(dict(original))
        for key in list(hidden):
            key_text = _normalized_key(key)
            if "cab" in key_text and any(token in key_text for token in ("outcome", "recommendation", "decision", "status", "result", "comment")):
                hidden.pop(key, None)
        hidden.pop("historical_prediction_label", None)
        examples.append(BenchmarkExample(hidden, actual, source_id(original)))
    return examples


def benchmark_diagnostics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    normal_rows = [row for row in rows if _is_normal(row)]
    outcomes = Counter(_outcome_text(row) for row in normal_rows if _outcome_text(row))
    return {
        "total_records": len(rows),
        "normal_records": len(normal_rows),
        "records_with_outcome_text": sum(bool(_outcome_text(row)) for row in normal_rows),
        "scorable_records": sum(normalize_outcome(row) is not None for row in normal_rows),
        "unscorable_records": sum(normalize_outcome(row) is None for row in normal_rows),
        "candidate_outcome_values": dict(outcomes.most_common(20)),
    }
