from __future__ import annotations
from collections import Counter
from typing import Any, Iterable
from .schemas import Decision


def normalize_cab_recommendation(value: Any) -> Decision | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in ("cancellation requested", "request for cancellation", "cancelled", "cancel", "reject", "insufficient information")):
        return Decision.NOT_READY
    if any(term in text for term in ("conditionally approved", "conditional", "condition", "test results reqd", "schedule change is required")):
        return Decision.CONDITIONAL
    if any(term in text for term in ("approved", "approve", "approvd")):
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
