"""Leakage-control policy for historical Pre-CAB benchmarking.

The historical ServiceNow export contains fields that are populated only after execution,
closure, or CAB review. Those fields must not be visible to the model during prediction.
"""
from __future__ import annotations

from typing import Any, Iterable

# Conservative deny-list: when in doubt, keep a field out of the historical benchmark input.
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


def strip_post_decision_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Return only fields considered safe as model-visible historical input."""
    return {key: value for key, value in record.items() if key not in POST_DECISION_FIELDS}


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
