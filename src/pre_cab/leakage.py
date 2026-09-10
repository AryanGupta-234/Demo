"""Leakage detection for historical CR benchmarking.

The benchmark must not expose fields that are only known because CAB already reviewed or
closed the change. This module is deliberately conservative: suspicious fields are flagged
for human review instead of silently being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


POST_DECISION_HINTS = (
    "cab outcome",
    "cab recommendation",
    "close code",
    "close notes",
    "closed",
    "closed by",
    "actual start date",
    "actual end date",
    "phase state",
    "approval history",
)


@dataclass(frozen=True)
class LeakageFinding:
    field: str
    reason: str
    action: str = "REVIEW"


def detect_leakage(fields: Iterable[str]) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    for field in fields:
        normalized = str(field).strip().lower()
        if any(hint in normalized for hint in POST_DECISION_HINTS):
            findings.append(
                LeakageFinding(
                    field=str(field),
                    reason="Field can contain information created or changed after CAB review.",
                )
            )
    return findings


def sanitize_model_record(record: dict) -> tuple[dict, list[LeakageFinding]]:
    findings = detect_leakage(record.keys())
    blocked = {f.field for f in findings}
    return {k: v for k, v in record.items() if k not in blocked}, findings
