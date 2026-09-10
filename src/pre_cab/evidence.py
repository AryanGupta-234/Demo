"""Stage-2 evidence verification over extracted attachment text/metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness


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
    if _contains_any(low, ("expected result", "actual result", "pass", "passed", "test case", "test cases", "execution result")):
        return "strong"
    if _contains_any(low, ("tested", "testing completed", "testing", "uat completed", "uat", "test result")):
        return "medium"
    return "weak"


def _severity_for_weak_evidence(strictness: Strictness) -> FindingSeverity:
    if strictness == Strictness.STRICT:
        return FindingSeverity.BLOCKING
    if strictness == Strictness.BALANCED:
        return FindingSeverity.WARNING
    return FindingSeverity.INFO


def verify_attachments(
    cr: dict[str, Any],
    documents: list[EvidenceDocument],
    requirements: list[Requirement] | None = None,
    strictness: Strictness = Strictness.BALANCED,
) -> EvidenceResult:
    """Verify CR claims against extracted attachments without inventing evidence."""
    cr_number = str(cr.get("Number") or cr.get("Effective number") or "").strip()
    relevant = [d for d in documents if _mentions_cr(d, cr_number)] or documents
    findings: list[Finding] = []
    verified: dict[str, bool] = {}
    contradictions: list[str] = []

    reqs = requirements or []
    test_required = any(r.required and r.name.lower() in {"uat", "testing", "test evidence"} for r in reqs)
    test_claim = _norm(cr.get("Test Results Evidence")) in {"yes", "available", "attached"} or bool(str(cr.get("Test plan") or "").strip())
    test_docs = [
        d for d in relevant
        if _contains_any(
            d.text,
            (
                "uat", "test result", "test case", "test evidence", "user acceptance",
                "execution result", "testing", "tested", "expected result", "actual result",
            ),
        )
    ]
    verified["testing"] = bool(test_docs) if (test_required or test_claim) else True
    if test_required or test_claim:
        if not test_docs:
            findings.append(Finding(
                "EVIDENCE_TEST_MISSING", "Testing evidence not verified", FindingSeverity.BLOCKING,
                "The CR claims or requires testing, but no supporting testing document was found.",
                recommendation="Attach the relevant test execution evidence.",
            ))
        else:
            strength = max((_evidence_strength(d.text) for d in test_docs), key={"weak": 0, "medium": 1, "strong": 2}.get)
            severity = FindingSeverity.INFO if strength == "strong" else _severity_for_weak_evidence(strictness)
            title = "Testing evidence verified" if strength == "strong" else "Testing evidence needs stronger detail"
            message = "Supporting testing documentation was found." if strength == "strong" else "Testing is referenced, but the attachment contains limited execution detail."
            recommendation = "" if strength == "strong" else "Provide test cases and expected/actual results where applicable."
            findings.append(Finding(
                "EVIDENCE_TEST_VERIFIED" if strength == "strong" else "EVIDENCE_TEST_WEAK",
                title,
                severity,
                message,
                technical_detail=f"Evidence strength={strength}; documents={[d.ref for d in test_docs]}",
                evidence_refs=tuple(d.ref for d in test_docs),
                recommendation=recommendation,
            ))

    customer_required = any(r.required and r.name.lower() == "customer approval" for r in reqs)
    customer_claim = _norm(cr.get("Customer Approval")) in {"yes", "approved"}
    approval_docs = [d for d in relevant if _contains_any(d.text, ("customer approval", "customer approved", "approved by customer", "customer accepted", "we approve"))]
    verified["customer_approval"] = bool(approval_docs) if (customer_required or customer_claim) else True
    if customer_required or customer_claim:
        if not approval_docs:
            findings.append(Finding(
                "EVIDENCE_CUSTOMER_APPROVAL_MISSING", "Customer approval not verified", FindingSeverity.BLOCKING,
                "The change requires or claims customer approval, but no matching approval evidence was found.",
                recommendation="Attach the customer approval evidence or document an approved exception.",
            ))
        else:
            findings.append(Finding(
                "EVIDENCE_CUSTOMER_APPROVAL_VERIFIED", "Customer approval evidence found", FindingSeverity.INFO,
                "Supporting customer-approval documentation was found.",
                evidence_refs=tuple(d.ref for d in approval_docs),
            ))

    backout = _norm(cr.get("Backout plan"))
    rollback_docs = [d for d in relevant if _contains_any(d.text, ("rollback", "backout", "restore", "revert", "backup", "recovery"))]
    verified["rollback"] = bool(backout) or bool(rollback_docs)
    if backout and rollback_docs:
        findings.append(Finding(
            "EVIDENCE_ROLLBACK_CORROBORATED", "Rollback approach corroborated", FindingSeverity.INFO,
            "The CR rollback statement is supported by attachment content.",
            evidence_refs=tuple(d.ref for d in rollback_docs),
        ))
    elif backout:
        findings.append(Finding(
            "EVIDENCE_ROLLBACK_SELF_DECLARED", "Rollback is self-declared", FindingSeverity.INFO,
            "A rollback mechanism is present in the CR; no separate corroborating document was found.",
        ))
    else:
        findings.append(Finding(
            "EVIDENCE_ROLLBACK_MISSING", "Rollback evidence missing", FindingSeverity.BLOCKING,
            "No rollback/backout mechanism was found in the CR or attachments.",
            recommendation="Provide a workable recovery mechanism or an explicit documented exception.",
        ))

    claim_text = f"{cr.get('Test plan', '')} {cr.get('Comments and Work notes', '')}".lower()
    says_uat = "uat" in claim_text
    dev_only_docs = [
        d for d in test_docs
        if _contains_any(d.text, ("dev", "development")) and not _contains_any(d.text, ("uat", "user acceptance"))
    ]
    has_uat_doc = any(_contains_any(d.text, ("uat", "user acceptance")) for d in test_docs)
    if says_uat and dev_only_docs and not has_uat_doc:
        contradictions.append("Testing environment contradiction: CR references UAT, but available testing evidence references DEV only.")
        verified["testing"] = False
        findings.append(Finding(
            "EVIDENCE_TEST_ENV_MISMATCH", "Testing environment mismatch", FindingSeverity.BLOCKING,
            "The CR and available testing evidence describe different test environments.",
            evidence_refs=tuple(d.ref for d in dev_only_docs),
            recommendation="Resolve the environment mismatch and provide evidence for the claimed environment.",
        ))

    blocking = any(f.severity == FindingSeverity.BLOCKING for f in findings)
    warnings = any(f.severity == FindingSeverity.WARNING for f in findings)
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)
    confidence = 0.95 if decision == Decision.PASS else (0.88 if decision == Decision.CONDITIONAL else 0.92)
    return EvidenceResult(decision, confidence, findings, verified, contradictions, [d.ref for d in relevant])
