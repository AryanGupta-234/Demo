"""Mine advisory field signals from historical Normal CR outcomes.

These statistics are evidence for rule design, not governance policy. They help identify fields whose
presence/absence is associated with historical CAB outcomes while keeping explicit rules reviewable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .cab_outcomes import normalize_cab_recommendation
from .schemas import Decision


@dataclass(frozen=True)
class FieldSignal:
    field: str
    context: str
    rows: int
    populated_rows: int
    missing_rows: int
    pass_rate_when_populated: float
    pass_rate_when_missing: float
    conditional_rate_when_populated: float
    conditional_rate_when_missing: float
    not_ready_rate_when_populated: float
    not_ready_rate_when_missing: float
    pass_lift: float
    support: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _present(value: Any) -> bool:
    return value not in (None, "", [], {}) and str(value).strip().lower() not in {"none", "nan", "null"}


def _rate(labels: list[Decision], label: Decision) -> float:
    return sum(item == label for item in labels) / len(labels) if labels else 0.0


def _signal(field: str, rows: list[dict[str, Any]], context: str) -> FieldSignal | None:
    populated: list[Decision] = []
    missing: list[Decision] = []
    for row in rows:
        outcome = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
        if outcome is None:
            continue
        (populated if _present(row.get(field)) else missing).append(outcome)
    total = len(populated) + len(missing)
    if not total:
        return None
    populated_pass = _rate(populated, Decision.PASS)
    missing_pass = _rate(missing, Decision.PASS)
    return FieldSignal(
        field=field,
        context=context,
        rows=total,
        populated_rows=len(populated),
        missing_rows=len(missing),
        pass_rate_when_populated=populated_pass,
        pass_rate_when_missing=missing_pass,
        conditional_rate_when_populated=_rate(populated, Decision.CONDITIONAL),
        conditional_rate_when_missing=_rate(missing, Decision.CONDITIONAL),
        not_ready_rate_when_populated=_rate(populated, Decision.NOT_READY),
        not_ready_rate_when_missing=_rate(missing, Decision.NOT_READY),
        pass_lift=populated_pass - missing_pass,
        support=max(len(populated), len(missing)) / total,
    )


def mine_field_signals(
    records: Iterable[dict[str, Any]],
    *,
    fields: Iterable[str] | None = None,
    context_fields: tuple[str, ...] = ("Category", "Sub Category", "Change Class"),
    minimum_rows: int = 20,
) -> list[FieldSignal]:
    normal = [row for row in records if str(row.get("Type") or "").strip().lower() == "normal"]
    selected_fields = sorted(set(fields or {key for row in normal for key in row}))
    results: list[FieldSignal] = []

    for field in selected_fields:
        signal = _signal(field, normal, "ALL_NORMAL")
        if signal and signal.rows >= minimum_rows:
            results.append(signal)

    for context_field in context_fields:
        values = sorted({str(row.get(context_field) or "").strip() for row in normal if str(row.get(context_field) or "").strip()})
        for value in values:
            subset = [row for row in normal if str(row.get(context_field) or "").strip() == value]
            if len(subset) < minimum_rows:
                continue
            context = f"{context_field}={value}"
            for field in selected_fields:
                signal = _signal(field, subset, context)
                if signal and signal.rows >= minimum_rows:
                    results.append(signal)

    return sorted(results, key=lambda item: (abs(item.pass_lift), item.rows), reverse=True)
