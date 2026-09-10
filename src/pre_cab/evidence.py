"""Evidence verification for attachments and supporting change records."""
from __future__ import annotations

from typing import Any, Iterable

from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness


class EvidenceDocument:
    def __init__(self, ref: str, name: str, text: str, document_type: str = "unknown", metadata: dict[str, Any] | None = None) -> None:
        self.ref = ref
        self.name = name
        self.text = text
        self.document_type = document_type
        self.metadata = metadata or {}


class EvidenceResult:
    def __init__(self, decision: Decision, confidence: float, findings: list[Finding], verified: dict[str, bool], contradictions: list[str], documents_considered: list[str]) -> None:
        self.decision = decision
        self.confidence = confidence
        self.findings = findings
        self.verified = verified
        self.contradictions = contradictions
        self.documents_considered = documents_considered


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = _norm(text)
    return any(term in normalized for term in terms)


def _evidence_strength(text: str) -> str:
    normalized = _norm(text)
    strong = ("expected result" in normalized and "actual result" in normalized) or "pass" in normalized
    medium = any(term in normalized for term in ("test case", "test result", "execution result", "uat", "user acceptance"))
    return "strong" if strong else "medium" if medium else "weak"


def _severity_for_weak_evidence(strictness: Strictness) -> FindingSeverity:
    return FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING


def _relevant_documents(documents: list[EvidenceDocument]) -> list[EvidenceDocument]:
    return documents


def verify_attachments(
    cr: dict[str, Any],
    documents: list[EvidenceDocument],
    *,
    requirements: list[Requirement] | None = None,
    strictness: Strictness = Strictness.BALANCED,
) -> EvidenceResult:
    relevant = _relevant_documents(documents)
    findings: list[Finding] = []
    verified: dict[str, bool] = {}
    contradictions: list[str] = []

    reqs = requirements or []
    test_required = any(r.required and r.name.lower() in {"uat", "testing", "test evidence"} for r in reqs)
    test_claim = _norm(cr.get("Test Results Evidence")) in {"yes", "available", "attached"} or bool(str(cr.get("Test plan") or "").strip())
    test_docs = [d for d in relevant if _contains_any(d.text, ("uat", "test result", "test case", "test evidence", "user acceptance", "execution result", "testing", "tested", "expected result", "actual result"))]
    verified["testing"] = bool(test_docs) if (test_required or test_claim) else True
    if test_required or test_claim:
        if not test_docs:
            findings.append(Finding("EVIDENCE_TEST_MISSING", "Testing evidence not verified", FindingSeverity.BLOCKING, "The CR claims or requires testing, but no supporting testing document was found.", recommendation="Attach the relevant test execution evidence."))
        else:
            strength_rank = {"weak": 0, "medium": 1, "strong": 2}
            strength = max(((_evidence_strength(d.text), d) for d in test_docs), key=lambda item: strength_rank[item[0]])[0]
            severity = FindingSeverity.INFO if strength == "strong" else _severity_for_weak_evidence(strictness)
            findings.append(Finding("EVIDENCE_TEST_VERIFIED" if strength == "strong" else "EVIDENCE_TEST_WEAK", "Testing evidence verified" if strength == "strong" else "Testing evidence needs stronger detail", severity, "Supporting testing documentation was found." if strength == "strong" else "Testing is referenced, but the attachment contains limited execution detail.", technical_detail=f"Evidence strength={strength}; documents={[d.ref for d in test_docs]}", evidence_refs=tuple(d.ref for d in test_docs), recommendation="" if strength == "strong" else "Provide test cases and expected/actual results where applicable."))

    approval_required = any(r.required and r.name.lower() in {"customer approval", "approval"} for r in reqs)
    approval_claim = _norm(cr.get("Customer Approval")) in {"yes", "approved", "available", "attached"}
    approval_docs = [d for d in relevant if _contains_any(d.text, ("customer approval", "approved by customer", "customer approved", "approval"))]
    verified["customer_approval"] = bool(approval_docs) if approval_required or approval_claim else True
    if approval_required or approval_claim:
        if not approval_docs:
            findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_MISSING", "Customer approval not verified", FindingSeverity.BLOCKING, "The CR claims or requires customer approval, but no supporting approval evidence was found.", recommendation="Attach the customer approval evidence."))
        else:
            findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_VERIFIED", "Customer approval verified", FindingSeverity.INFO, "Supporting customer approval evidence was found.", evidence_refs=tuple(d.ref for d in approval_docs)))

    rollback_claim = _norm(cr.get("Backout plan"))
    rollback_docs = [d for d in relevant if _contains_any(d.text, ("rollback", "backout", "restore", "recovery"))]
    verified["rollback"] = bool(rollback_docs) if rollback_claim else True
    if rollback_claim and rollback_docs:
        findings.append(Finding("EVIDENCE_ROLLBACK_CORROBORATED", "Rollback evidence corroborated", FindingSeverity.INFO, "Supporting rollback or recovery detail was found in the evidence.", evidence_refs=tuple(d.ref for d in rollback_docs)))
    elif rollback_claim:
        findings.append(Finding("EVIDENCE_ROLLBACK_SELF_DECLARED", "Rollback evidence is self-declared", FindingSeverity.WARNING, "The CR contains a rollback plan, but no attachment independently corroborates it."))
    else:
        findings.append(Finding("EVIDENCE_ROLLBACK_MISSING", "Rollback plan missing", FindingSeverity.BLOCKING, "No rollback/backout plan is present on the CR.", recommendation="Provide a concrete rollback or recovery plan."))

    for document in relevant:
        if document.metadata.get("extraction_error"):
            findings.append(Finding("EVIDENCE_UNREADABLE", "Evidence could not be fully parsed", FindingSeverity.WARNING, f"Attachment {document.name} could not be extracted: {document.metadata['extraction_error']}", evidence_refs=(document.ref,)))
        if document.metadata.get("requires_vision") and not document.text.strip():
            findings.append(Finding("EVIDENCE_UNREADABLE", "Image evidence requires visual review", FindingSeverity.WARNING, f"Attachment {document.name} is image-based and has no extracted text.", evidence_refs=(document.ref,)))

    dev_only = any("dev-only" in _norm(d.text) or "dev testing" in _norm(d.text) for d in relevant)
    prod_target = _norm(cr.get("Environment")) in {"prod", "production"}
    if dev_only and (prod_target or test_required):
        contradictions.append("Attachment indicates DEV-only validation while the CR requires UAT or targets production.")
        findings.append(Finding("EVIDENCE_TEST_ENV_MISMATCH", "Testing environment mismatch", FindingSeverity.BLOCKING, "Evidence indicates DEV-only validation while the CR requires UAT or targets production.", recommendation="Provide UAT/production-equivalent validation evidence or resolve the environment mismatch before CAB."))

    if any(f.severity == FindingSeverity.BLOCKING for f in findings):
        decision = Decision.NOT_READY
    elif any(f.severity == FindingSeverity.WARNING for f in findings):
        decision = Decision.CONDITIONAL
    else:
        decision = Decision.PASS
    confidence = 0.98 if decision == Decision.PASS else 0.90 if decision == Decision.CONDITIONAL else 0.85
    return EvidenceResult(decision, confidence, findings, verified, contradictions, [d.ref for d in relevant])
