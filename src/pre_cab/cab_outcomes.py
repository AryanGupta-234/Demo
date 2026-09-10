from __future__ import annotations
from collections import Counter
from typing import Any, Iterable
from .schemas import Decision


def normalize_cab_recommendation(value: Any) -> Decision | None:
    """Map only reasonably explicit CAB outcomes; requests/instructions remain unscored."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in (
        "cancellation requested", "request for cancellation", "cancelled", "canceled",
        "cancel as", "not approved", "rejected", "reject", "hold", "insufficient information",
    )):
        return Decision.NOT_READY
    if any(term in text for term in (
        "conditionally approved", "conditional", "condition", "schedule needs to be updated",
        "schedule change is required", "subject to", "test results reqd",
        "customer approval reqd", "customer approval required",
    )):
        return Decision.CONDITIONAL
    if any(term in text for term in (
        "approved", "change is approved", "cr is approved", "approval granted", "approval attached",
    )) and not any(term in text for term in ("please approve", "pls approve", "request approval")):
        return Decision.PASS
    return None


def outcome_distribution(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        if str(record.get("Type") or "").strip().lower() != "normal":
            continue
        label = normalize_cab_recommendation(record.get("CAB recommendation"))
        counter[label.value if label else "UNSCORABLE"] += 1
    return dict(counter)
