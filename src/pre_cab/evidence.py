"""Stage-2 evidence verification over extracted attachment text/metadata.

The adapter intentionally works on extracted document records so PDF/XLSX/email parsers can be
plugged in later without changing validation logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import re

from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness, ValidationResult


@dataclass(frozen=True)
class EvidenceDocument:
    ref: str
    name: str
    text: str
    document_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceResult:
    decision: Decision
    confidence: float
    findings: list[Finding]
    verified: dict[str, bool]
    contradictions: list[str]
    documents_considered: list[str]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _mentions_cr(doc: EvidenceDocument, cr_number: str) -> bool:
    if not cr_number:
        return True
    return cr_number.lower() in doc.text.lower() or cr_number.lower() in doc.name.lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def _evidence_strength(text: str) -> str:
    low = text.lower()
    if _contains_any(low, ["expected result", "actual result", "pass", "passed", "test case", "test cases"]):
        return "strong"
    if _contains_any(low, ["tested", "testing completed", "uat completed", "uat"]):
        return "medium"
    return "weak"


def verify_attachments(
    cr: dict[str, Any],
    documents: list[EvidenceDocument],
    requirements: list[Requirement] | None = None,
    strictness: Strictness = Strictness.BALANCED,
) -> EvidenceResult:
    """Verify claims made by a CR against extracted attachment content.

    This stage never invents evidence. A document only supports a claim when its content is
    discoverable and, where possible, tied to the current CR number.
    """
    cr_number = str(cr.get("Number") or cr.get("Effective number") or "").strip()
    relevant = [d for d in documents if _mentions_cr(d, cr_number)] or documents
    findings: list[Finding] = []
    verified: dict[str, bool] = {}
    contradictions: list[str] = []

    test_required = any(r.required and r.name.lower() in {"uat", "testing", "test evidence"} for r in (requirements or []))
    test_claim = _norm(cr.get("Test Results Evidence")) in {"yes", "available", "attached"} or bool(str(cr.get("Test plan") or "").strip())
    test_docs = [d for d in relevant if _contains_any(d.text, ["uat", "test result", "test case", "test evidence", "user acceptance"])]
    verified["testing"] = bool(test_docs) if (test_required or test_claim) else True
    if test_required or test_claim:
        if test_docs:
            strength = max((_evidence_strength(d.text) for d in test_docs), key={"weak": 0, "medium": 1, "strong": 2}.get)
            if strength == "strong":
                findings.append(Finding("EVIDENCE_TEST_VERIFIED", "Testing evidence verified", FindingSeverity.INFO,
                    "Supporting testing documentation was found.", technical_detail=f"Documents: {[d.ref for d in test_docs]}", evidence_refs=tuple(d.ref for d in test_docs)))
            else:
                findings.append(Finding("EVIDENCE_TEST_WEAK", "Testing evidence is weak", FindingSeverity.WARNING,
                    "Testing is referenced, but the attachment contains limited execution detail.", technical_detail=f"Evidence strength={strength}", evidence_refs=tuple(d.ref for d in test_docs), recommendation="Provide test cases and expected/actual results where applicable."))
        else:
            findings.append(Finding("EVIDENCE_TEST_MISSING", "Testing evidence not verified", FindingSeverity.BLOCKING,
                "The CR claims or requires testing, but no supporting testing document was found.", recommendation="Attach the relevant test execution evidence."))

    customer_claim = _norm(cr.get("Customer Approval")) in {"yes", "approved"}
    approval_docs = [d for d in relevant if _contains_any(d.text, ["customer approval", "customer approved", "approved by", "approval"])]
    verified["customer_approval"] = bool(approval_docs) if customer_claim else True
    if customer_claim and not approval_docs:
        findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_MISSING", "Customer approval not verified", FindingSeverity.BLOCKING,
            "The CR records customer approval, but no matching approval evidence was found.", recommendation="Attach the customer approval evidence."))
    elif customer_claim:
        findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_VERIFIED", "Customer approval evidence found", FindingSeverity.INFO,
            "Supporting customer-approval documentation was found.", evidence_refs=tuple(d.ref for d in approval_docs)))

    backout = _norm(cr.get("Backout plan"))
    rollback_docs = [d for d in relevant if _contains_any(d.text, ["rollback", "backout", "restore", "revert", "backup"])]
    verified["rollback"] = bool(backout) or bool(rollback_docs)
    if backout and rollback_docs:
        findings.append(Finding("EVIDENCE_ROLLBACK_CORROBORATED", "Rollback approach corroborated", FindingSeverity.INFO,
            "The CR rollback statement is supported by attachment content.", evidence_refs=tuple(d.ref for d in rollback_docs)))
    elif backout:
        findings.append(Finding("EVIDENCE_ROLLBACK_SELF_DECLARED", "Rollback is self-declared", FindingSeverity.INFO,
            "A rollback mechanism is present in the CR; no separate corroborating document was found."))
    else:
        findings.append(Finding("EVIDENCE_ROLLBACK_MISSING", "Rollback evidence missing", FindingSeverity.BLOCKING,
            "No rollback/backout mechanism was found in the CR or attachments.", recommendation="Provide a workable recovery mechanism or an explicit documented exception."))

    # Detect a common high-value contradiction: CR claims UAT while an evidence document says DEV only.
    claim_text = f"{cr.get('Test plan','')} {cr.get('Comments and Work notes','')}".lower()
    says_uat = "uat" in claim_text
    dev_only = any("dev" in d.text.lower() and "uat" not in d.text.lower() for d in test_docs)
    if says_uat and dev_only:
        contradictions.append("CR references UAT, but a supporting testing document appears to reference DEV only.")
        findings.append(Finding("EVIDENCE_TEST_ENV_MISMATCH", "Testing environment mismatch", FindingSeverity.BLOCKING,
            "The CR and supporting testing evidence describe different test environments.", recommendation="Resolve the environment mismatch and provide evidence for the claimed environment."))

    blocking = any(f.severity == FindingSeverity.BLOCKING for f in findings)
    warnings = any(f.severity == FindingSeverity.WARNING for f in findings)
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)
    confidence = 0.95 if decision == Decision.PASS else (0.88 if decision == Decision.CONDITIONAL else 0.92)
    return EvidenceResult(decision, confidence, findings, verified, contradictions, [d.ref for d in relevant])
