"""Prepare historical Normal CRs for leakage-safe benchmark replay."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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
        if normalized in {"normal", "normal change", "normalchange"} or "normal change" in normalized:
            return True
    return False


_DECISION_TERMS: tuple[str, ...] = (
    "approved", "approve", "approvd", "conditionally approved", "conditional", "rejected",
    "reject", "not approved", "not ready", "hold", "held", "pending approval", "cancelled",
    "canceled", "cancellation", "go ahead", "okay to proceed", "test results reqd",
    "schedule needs to be updated",
)


def _looks_like_outcome_field(key: object) -> bool:
    normalized = _normalized_key(key)
    return any(token in normalized for token in (
        "cab", "outcome", "recommendation", "decision", "approval", "review", "result",
        "disposition", "action", "comment", "remark", "note",
    ))


def discover_outcome_fields(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank likely historical decision fields by field name and decision vocabulary."""
    normal_rows = [row for row in records if _is_normal(row)]
    fields: list[dict[str, Any]] = []
    keys = sorted({str(key) for row in normal_rows for key in row if _looks_like_outcome_field(key)})
    for key in keys:
        values = [str(row.get(key) or "").strip() for row in normal_rows]
        nonempty = [value for value in values if value]
        decision_hits = sum(any(term in value.lower() for term in _DECISION_TERMS) for value in nonempty)
        key_text = _normalized_key(key)
        name_score = sum(token in key_text for token in ("cab", "outcome", "recommendation", "decision", "disposition"))
        decision_rate = decision_hits / len(nonempty) if nonempty else 0.0
        score = decision_rate * 100 + name_score * 10
        if nonempty:
            fields.append({
                "field": key,
                "nonempty": len(nonempty),
                "decision_hits": decision_hits,
                "decision_rate": round(decision_rate, 4),
                "score": round(score, 2),
                "sample_values": list(dict.fromkeys(nonempty))[:10],
            })
    fields.sort(key=lambda item: (item["score"], item["decision_rate"], item["nonempty"]), reverse=True)
    return fields


def _outcome_field(record: dict[str, Any], preferred_field: str | None = None) -> str:
    if preferred_field:
        for key, value in record.items():
            if str(key) == preferred_field:
                return str(value or "").strip().lower()
    direct = _field_text(
        record,
        "CAB Outcome", "CAB recommendation", "CAB Recommendation", "cab_outcome", "cab_recommendation",
        "CAB Decision", "CAB Status", "Final CAB Decision", "Final CAB Outcome",
        "CAB Decision/Recommendation", "CAB Comments", "CAB Comment",
    )
    if direct:
        return direct
    candidates = discover_outcome_fields([record])
    return str(record.get(candidates[0]["field"]) or "").strip().lower() if candidates else ""


def normalize_outcome(record: dict[str, Any], preferred_field: str | None = None) -> Decision | None:
    """Map explicit historical decision text to a three-way benchmark label."""
    raw = _outcome_field(record, preferred_field)
    if not raw:
        return None
    if any(term in raw for term in (
        "cancellation requested", "request for cancellation", "cancelled as", "cancelled", "canceled",
        "cancel", "rejected", "reject", "insufficient information", "not ready", "not approved",
        "hold", "held", "do not approve", "don't approve",
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


def prepare_normal_benchmark(records: Iterable[dict[str, Any]], preferred_outcome_field: str | None = None) -> list[BenchmarkExample]:
    rows = list(records)
    preferred = preferred_outcome_field or (discover_outcome_fields(rows)[0]["field"] if discover_outcome_fields(rows) else None)
    examples: list[BenchmarkExample] = []
    for original in rows:
        if not _is_normal(original):
            continue
        actual = normalize_outcome(original, preferred)
        hidden = strip_post_decision_fields(dict(original))
        for key in list(hidden):
            if _looks_like_outcome_field(key):
                hidden.pop(key, None)
        hidden.pop("historical_prediction_label", None)
        examples.append(BenchmarkExample(hidden, actual, source_id(original)))
    return examples


def benchmark_diagnostics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    normal_rows = [row for row in rows if _is_normal(row)]
    candidates = discover_outcome_fields(rows)
    preferred = candidates[0]["field"] if candidates and candidates[0]["decision_hits"] else None
    outcomes = Counter(_outcome_field(row, preferred) for row in normal_rows if _outcome_field(row, preferred))
    return {
        "total_records": len(rows),
        "normal_records": len(normal_rows),
        "candidate_outcome_fields": candidates[:15],
        "selected_outcome_field": preferred,
        "records_with_outcome_text": sum(bool(_outcome_field(row, preferred)) for row in normal_rows),
        "scorable_records": sum(normalize_outcome(row, preferred) is not None for row in normal_rows),
        "unscorable_records": sum(normalize_outcome(row, preferred) is None for row in normal_rows),
        "candidate_outcome_values": dict(outcomes.most_common(20)),
    }
