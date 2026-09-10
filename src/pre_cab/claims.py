"""Claim/evidence contracts for Stage-2 document verification."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClaimStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    category: str
    statement: str
    required: bool
    source_field: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvidenceMatch:
    claim_id: str
    document_ref: str
    status: ClaimStatus
    strength: float
    excerpt: str = ""
    reason: str = ""


def claim_for_uat(required: bool, source: str = "context") -> Claim:
    return Claim(
        claim_id="uat-required",
        category="testing",
        statement="Appropriate functional/UAT validation exists when this change requires it.",
        required=required,
        source_field=source,
    )


def claim_for_customer_approval(required: bool) -> Claim:
    return Claim(
        claim_id="customer-approval",
        category="approval",
        statement="Required customer approval exists and applies to this change.",
        required=required,
        source_field="Customer Approval",
    )


def claim_for_rollback() -> Claim:
    return Claim(
        claim_id="rollback",
        category="recovery",
        statement="A credible recovery/backout mechanism exists for the proposed change.",
        required=True,
        source_field="Backout plan",
    )
