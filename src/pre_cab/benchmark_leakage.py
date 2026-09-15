"""Leakage-control policy for historical Pre-CAB benchmarking.

The historical ServiceNow export contains fields that are populated only after execution,
closure, or CAB review. Those fields must not be visible to a leakage-sensitive benchmark.
The explicit learning pipeline may opt into same-CR Work Notes/Comments because they are part
of the user's requested historical learning corpus; those journals must then be treated as
auxiliary chronological evidence, never as labels.
"""
from __future__ import annotations

from typing import Any, Iterable

POST_DECISION_FIELDS: frozenset[str] = frozenset(
    {
        "CAB Outcome",
        "CAB recommendation",
        "CAB Recommendation",
        "CAB date",
        "CAB date/time",
        "CAB delegate",
        "State",
        "Phase",
        "Phase state",
        "Sub State",
        "Closed",
        "Closed by",
        "Close code",
        "Close notes",
        "Actual start date",
        "Actual end date",
        "Updated",
        "Updated by",
        "Updates",
        "Comments and Work notes",
        "Work notes",
        "Approval history",
        "Approval set",
        "Upon approval",
        "Upon reject",
        "Active",
    }
)

JOURNAL_FIELDS: frozenset[str] = frozenset({"Work notes", "Comments", "Comments and Work notes", "Notes"})


def strip_post_decision_fields(
    record: dict[str, Any],
    *,
    include_journals: bool = False,
) -> dict[str, Any]:
    """Remove post-decision fields; optionally preserve same-CR journal fields for learning."""
    blocked = POST_DECISION_FIELDS - JOURNAL_FIELDS if include_journals else POST_DECISION_FIELDS
    return {key: value for key, value in record.items() if key not in blocked}


def leakage_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report which post-decision fields occur in a collection for audit/debugging."""
    counts: dict[str, int] = {field: 0 for field in sorted(POST_DECISION_FIELDS)}
    total = 0
    for record in records:
        total += 1
        for field in POST_DECISION_FIELDS:
            if field in record:
                counts[field] += 1
    return {
        "records": total,
        "protected_fields": len(POST_DECISION_FIELDS),
        "present_counts": {k: v for k, v in counts.items() if v},
    }
