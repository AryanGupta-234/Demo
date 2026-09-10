from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .schemas import Decision


def normalize_cab_recommendation(value: Any) -> Decision | None:
    """Normalize messy historical CAB text conservatively.

    Approval requests, attachment notices, and ambiguous schedule-only notes remain unscored.
    """
    text = str(value or "").strip().lower()
    if not text:
        return None

    # Hard negative outcomes.
    if any(term in text for term in (
        "cancellation requested", "request for cancellation", "cancelled as",
        "cancelled", "cancelled as requested", "cancel as", "rejected",
        "reject", "insufficient information", "not approved", "not ready", "held",
    )):
        return Decision.NOT_READY

    # Explicit conditional outcomes or approvals with unresolved requirements.
    if any(term in text for term in (
        "conditionally approved", "conditionallyapprove", "conditional",
        "condition", "test results reqd", "schedule change is required",
        "schedule needs to be updated", "customer approval pending",
        "approval pending", "approval is pending", "implementation is not permitted",
        "reschedule request", "schedule change required",
    )):
        return Decision.CONDITIONAL

    # Clearly approved outcomes. Accept common spelling/punctuation variants only when
    # the sentence is approval-dominant and does not contain unresolved requirements.
    compact = " ".join(text.replace("\n", " ").split())
    approval_like = compact in {
        "approved", "approved.", "approve", "approvd", "apprvd", "apoved",
        "aproved", "approved'", "cab approved.", "cab approved", "appoved.",
    } or compact.startswith("approved ")
    if approval_like and not any(term in compact for term in (
        "pending", "reqd", "required", "not permitted", "schedule change",
        "customer approval", "test result",
    )):
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
