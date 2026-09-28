"""Evidence verification for attachments and supporting change records."""
from __future__ import annotations

import re
from typing import Any, Iterable

from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness
from .evidence_agents import DEFAULT_DOCUMENT_AGENTS


class EvidenceDocument:
    def __init__(self, ref: str, name: str, text: str, document_type: str = "unknown", metadata: dict[str, Any] | None = None) -> None:
        self.ref = ref
        self.name = name
        self.text = text
        self.document_type = document_type
        self.metadata = metadata or {}


class EvidenceResult:
    def __init__(self, decision: Decision, confidence: float, findings: list[Finding], verified: dict[str, Any], contradictions: list[str], documents_considered: list[str]) -> None:
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
    return any(term.lower() in normalized for term in terms)


def _doc_matches(document: EvidenceDocument, kind: str) -> bool:
    classification = document.metadata.get("evidence_classification") or {}
    if classification.get("type") == kind:
        return True
    haystack = f"{document.name}\n{document.text[:16000]}".lower()
    terms = {
        "customer_approval": ("customer approval", "customer_approval", "customer-approval", "approved by customer", "customer approved"),
        "uat": ("uat", "user acceptance", "acceptance signoff"),
        "qa_signoff": ("tcs qa", "qa signoff", "quality assurance", "qa approval"),
        "test_results": ("test result", "test execution", "test evidence", "test report", "execution report", "expected result", "actual result"),
        "lower_environment": ("lower environment", "pre-prod", "preprod", "sit", "staging", "lower env"),
        "rollback": ("rollback", "backout", "restore", "recovery", "revert"),
    }
    return any(term in haystack for term in terms.get(kind, ()))


def _has_positive_approval(text: str) -> bool:
    normalized = _norm(text)
    if any(term in normalized for term in ("not approved", "rejected", "declined", "approval pending", "awaiting approval", "pending approval")):
        return False
    return any(term in normalized for term in ("approved", "approval granted", "approved by customer", "customer approved", "signoff approved", "accepted"))


def _has_positive_uat(text: str) -> bool:
    normalized = _norm(text)
    if any(term in normalized for term in ("uat failed", "uat fail", "user acceptance failed", "rejected")):
        return False
    return any(term in normalized for term in ("uat passed", "uat pass", "uat successful", "uat completed", "user acceptance passed", "user acceptance complete", "accepted", "business sign-off", "business signoff"))


def _test_result_state(text: str) -> str:
    """Resolve the most recent explicit execution state instead of any historical word hit."""
    normalized = _norm(text)
    patterns = (
        ("failed", r"\b(?:test|testing|execution|validation)(?:\s+result)?\s*(?:is|was|:)?\s*(?:failed|failure|not passed)\b"),
        ("passed", r"\b(?:test|testing|execution|validation)(?:\s+result)?\s*(?:is|was|:)?\s*(?:passed|pass|successful|success)\b"),
        ("failed", r"\b(?:failed|not passed|execution failed|testing failed)\b"),
        ("passed", r"\b(?:all tests? passed|tests passed|test cases passed|execution passed)\b"),
    )
    matches: list[tuple[int, str]] = []
    for state, pattern in patterns:
        matches.extend((match.start(), state) for match in re.finditer(pattern, normalized))
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    if "expected result" in normalized and "actual result" in normalized:
        # Expected/actual fields establish test evidence, but not pass/fail by themselves.
        actual = re.search(r"actual result\s*[:=-]\s*([^\n|;]+)", normalized)
        if actual:
            value = actual.group(1)
            if re.search(r"\b(pass|passed|successful|success|ok)\b", value):
                return "passed"
            if re.search(r"\b(fail|failed|failure|not passed)\b", value):
                return "failed"
        return "present"
    if any(term in normalized for term in ("test case", "test result", "execution result", "tested", "validated")):
        return "present"
    return "unknown"


def _cr_identity_match(cr: dict[str, Any], document: EvidenceDocument) -> bool:
    number = str(cr.get("Number") or cr.get("Effective number") or "").strip().lower()
    if not number:
        return True
    return number in _norm(document.text) or number in _norm(document.name)


def _severity_for_weak_evidence(strictness: Strictness) -> FindingSeverity:
    return FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING


def verify_attachments(
    cr: dict[str, Any],
    documents: list[EvidenceDocument],
    *,
    requirements: list[Requirement] | None = None,
    strictness: Strictness = Strictness.BALANCED,
) -> EvidenceResult:
    findings: list[Finding] = []
    verified: dict[str, Any] = {}
    contradictions: list[str] = []
    reqs = requirements or []

    def required(name: str, *aliases: str) -> bool:
        wanted = {name.lower(), *(alias.lower() for alias in aliases)}
        return any(r.required and r.name.lower() in wanted for r in reqs)

    specialist = [agent.run(cr, documents) for agent in DEFAULT_DOCUMENT_AGENTS]
    verified["document_agents"] = {item["purpose"]: item for item in specialist}

    test_required = required("UAT", "testing", "test evidence") or bool(_norm(cr.get("Test plan")))
    test_documents = [d for d in documents if _doc_matches(d, "test_results") or _doc_matches(d, "uat")]
    test_identity_ok = [d for d in test_documents if _cr_identity_match(cr, d)]
    verified["testing"] = bool(test_identity_ok) if test_required else True
    if test_required:
        if not test_identity_ok:
            findings.append(Finding("EVIDENCE_TEST_MISSING", "Testing evidence not verified", FindingSeverity.BLOCKING, "Testing is required or claimed, but no CR-matching UAT/test document was found.", recommendation="Attach a CR-specific UAT or test execution result and ensure the CR number is identifiable."))
        else:
            states = [_test_result_state(d.text) for d in test_identity_ok]
            if "failed" in states:
                findings.append(Finding("EVIDENCE_TEST_FAILED", "Testing evidence records a failure", FindingSeverity.BLOCKING, "A CR-matching test document contains failed/not-passed execution evidence.", evidence_refs=tuple(d.ref for d in test_identity_ok), recommendation="Resolve the failed test execution and record a successful/reconciled result."))
            elif "passed" in states:
                findings.append(Finding("EVIDENCE_TEST_VERIFIED", "Testing evidence verified", FindingSeverity.INFO, "A CR-matching test document contains positive execution evidence.", evidence_refs=tuple(d.ref for d in test_identity_ok)))
            else:
                findings.append(Finding("EVIDENCE_TEST_WEAK", "Testing evidence is present but not conclusive", _severity_for_weak_evidence(strictness), "A CR-matching testing document was found, but the extracted content does not establish a clear passed execution result.", evidence_refs=tuple(d.ref for d in test_identity_ok), recommendation="Provide explicit execution status and expected/actual results where applicable."))

    customer_required = required("Customer Approval", "approval") or bool(_norm(cr.get("Customer Approval")))
    customer_documents = [d for d in documents if _doc_matches(d, "customer_approval")]
    customer_identity_ok = [d for d in customer_documents if _cr_identity_match(cr, d)]
    verified["customer_approval"] = bool(customer_identity_ok) if customer_required else True
    if customer_required:
        if not customer_identity_ok:
            findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_MISSING", "Customer approval not verified", FindingSeverity.BLOCKING, "Customer approval is required/claimed, but no CR-matching approval document was found.", recommendation="Attach the customer approval evidence with an identifiable CR reference."))
        elif not any(_has_positive_approval(d.text) for d in customer_identity_ok):
            findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_NOT_POSITIVE", "Customer approval is not positively verified", FindingSeverity.BLOCKING, "Approval-related evidence was found, but its extracted content does not establish a positive customer approval.", evidence_refs=tuple(d.ref for d in customer_identity_ok), recommendation="Provide an approval record that clearly states approval/acceptance and identifies the approver."))
        else:
            findings.append(Finding("EVIDENCE_CUSTOMER_APPROVAL_VERIFIED", "Customer approval verified", FindingSeverity.INFO, "A CR-matching document contains positive customer approval language.", evidence_refs=tuple(d.ref for d in customer_identity_ok)))

    uat_required = required("UAT") or "uat" in _norm(cr.get("Test plan")) or "user acceptance" in _norm(cr.get("Test plan"))
    if uat_required:
        uat_documents = [d for d in documents if _doc_matches(d, "uat")]
        uat_identity_ok = [d for d in uat_documents if _cr_identity_match(cr, d)]
        verified["uat"] = bool(uat_identity_ok)
        if not uat_identity_ok:
            findings.append(Finding("EVIDENCE_UAT_MISSING", "UAT evidence not verified", FindingSeverity.BLOCKING, "UAT is required by the change context, but no CR-matching UAT evidence was found.", recommendation="Provide UAT signoff/results tied to this CR."))
        elif not any(_has_positive_uat(d.text) for d in uat_identity_ok):
            findings.append(Finding("EVIDENCE_UAT_NOT_POSITIVE", "UAT evidence is not positively verified", FindingSeverity.BLOCKING, "A CR-matching UAT document exists, but its extracted content does not establish completed positive acceptance.", evidence_refs=tuple(d.ref for d in uat_identity_ok), recommendation="Provide explicit UAT pass/acceptance evidence."))
        else:
            findings.append(Finding("EVIDENCE_UAT_VERIFIED", "UAT verified", FindingSeverity.INFO, "A CR-matching UAT document contains positive acceptance evidence.", evidence_refs=tuple(d.ref for d in uat_identity_ok)))

    qa_required = required("TCS QA signoff", "QA signoff")
    if qa_required:
        qa_documents = [d for d in documents if _doc_matches(d, "qa_signoff")]
        qa_identity_ok = [d for d in qa_documents if _cr_identity_match(cr, d)]
        verified["qa_signoff"] = bool(qa_identity_ok)
        if not qa_identity_ok:
            findings.append(Finding("EVIDENCE_QA_MISSING", "TCS QA evidence not verified", FindingSeverity.BLOCKING, "TCS QA signoff is required, but no CR-matching QA evidence was found.", recommendation="Provide the QA signoff/evidence tied to this CR."))
        elif not any(_has_positive_approval(d.text) or _has_positive_uat(d.text) for d in qa_identity_ok):
            findings.append(Finding("EVIDENCE_QA_NOT_POSITIVE", "TCS QA evidence is not positively verified", FindingSeverity.BLOCKING, "A QA document exists, but positive signoff is not established by its extracted text.", evidence_refs=tuple(d.ref for d in qa_identity_ok), recommendation="Provide explicit QA approval/completion evidence."))
        else:
            findings.append(Finding("EVIDENCE_QA_VERIFIED", "TCS QA evidence verified", FindingSeverity.INFO, "A CR-matching QA document contains positive signoff/completion language.", evidence_refs=tuple(d.ref for d in qa_identity_ok)))

    lower_ref = str(cr.get("Lower Environment Reference CR/SR") or "").strip()
    lower_required = required("Lower Environment Reference CR/SR", "Lower Environment Reference")
    lower_documents = [d for d in documents if _doc_matches(d, "lower_environment")]
    if lower_required or lower_ref:
        lower_identity_ok = [d for d in lower_documents if _cr_identity_match(cr, d)]
        verified["lower_environment"] = bool(lower_identity_ok)
        if not lower_identity_ok:
            findings.append(Finding("EVIDENCE_LOWER_ENV_MISSING", "Lower-environment evidence not verified", FindingSeverity.BLOCKING if lower_required else FindingSeverity.WARNING, "A lower-environment reference/evidence is expected, but no matching evidence document was found.", recommendation="Provide the referenced lower-environment CR/SR evidence or explicitly justify why it is not applicable."))
        elif lower_ref and lower_ref.lower() not in _norm(" ".join(d.text for d in lower_identity_ok)):
            findings.append(Finding("EVIDENCE_LOWER_ENV_REFERENCE_MISMATCH", "Lower-environment reference not corroborated", FindingSeverity.WARNING if strictness != Strictness.STRICT else FindingSeverity.BLOCKING, f"The CR references {lower_ref!r}, but the supplied evidence does not visibly corroborate that reference.", evidence_refs=tuple(d.ref for d in lower_identity_ok), recommendation="Provide evidence containing the referenced CR/SR number."))
        else:
            findings.append(Finding("EVIDENCE_LOWER_ENV_VERIFIED", "Lower-environment evidence verified", FindingSeverity.INFO, "Lower-environment evidence is present and the supplied reference is corroborated.", evidence_refs=tuple(d.ref for d in lower_identity_ok)))

    rollback_claim = _norm(cr.get("Backout plan"))
    rollback_documents = [d for d in documents if _doc_matches(d, "rollback")]
    if rollback_claim and rollback_documents:
        findings.append(Finding("EVIDENCE_ROLLBACK_CORROBORATED", "Rollback evidence corroborated", FindingSeverity.INFO, "Supporting rollback/recovery language was found in the evidence.", evidence_refs=tuple(d.ref for d in rollback_documents)))
    elif rollback_claim:
        findings.append(Finding("EVIDENCE_ROLLBACK_SELF_DECLARED", "Rollback remains self-declared", FindingSeverity.WARNING, "The CR contains a rollback plan, but no attachment independently corroborates it.", recommendation="Use implementation/runbook evidence or other operational documentation to corroborate recovery steps."))
    else:
        findings.append(Finding("EVIDENCE_ROLLBACK_MISSING", "Rollback plan missing", FindingSeverity.BLOCKING, "No rollback/backout plan is present on the CR.", recommendation="Provide a concrete rollback/recovery path before CAB."))

    for document in documents:
        if document.metadata.get("extraction_error"):
            findings.append(Finding("EVIDENCE_UNREADABLE", "Evidence could not be fully parsed", FindingSeverity.WARNING, f"Attachment {document.name} could not be extracted: {document.metadata['extraction_error']}", evidence_refs=(document.ref,)))
        if document.metadata.get("requires_vision") and not document.text.strip():
            findings.append(Finding(
                "EVIDENCE_UNREADABLE",
                "Evidence content could not be extracted",
                FindingSeverity.WARNING,
                f"Attachment {document.name} is image-based and contains no machine-readable text.",
                evidence_refs=(document.ref,),
                recommendation="Route the attachment to visual/OCR review before relying on it as evidence.",
            ))
            findings.append(Finding(
                "EVIDENCE_VISION_REVIEW_REQUIRED",
                "Image evidence requires visual review",
                FindingSeverity.WARNING,
                f"Attachment {document.name} is image-based and has no extracted text.",
                evidence_refs=(document.ref,),
                recommendation="Route image evidence to a vision-capable review step.",
            ))

    specialist_negatives = [
        item for item in specialist if int(item.get("negative_count", 0)) > 0
    ]
    for item in specialist_negatives:
        negative_refs = tuple(c["ref"] for c in item.get("candidates", []) if c.get("status") == "negative")
        findings.append(Finding(
            f"EVIDENCE_AGENT_NEGATIVE_{item['purpose'].upper()}",
            f"{item['purpose']} document contains negative evidence",
            FindingSeverity.BLOCKING,
            f"Specialist document review found negative evidence for {item['purpose']}.",
            evidence_refs=negative_refs,
            recommendation=f"Resolve the negative {item['purpose']} evidence before CAB.",
        ))

    all_text = "\n".join(d.text for d in documents)
    prod_target = _norm(cr.get("Environment")) in {"prod", "production"} or not _norm(cr.get("Environment"))
    evidence_lower = _norm(all_text)
    dev_environment = bool(re.search(
        r"\b(?:dev|development)(?:\s+environment)?\b", evidence_lower
    ))
    validation_action = bool(re.search(
        r"\b(?:test|tested|testing|validate|validated|validation|verification|verified)\b",
        evidence_lower,
    ))
    if prod_target and dev_environment and validation_action:
        contradictions.append("Evidence indicates DEV environment validation while the change is targeted to PROD.")
        findings.append(Finding("EVIDENCE_PROD_DEV_CONTRADICTION", "PROD target with DEV-only evidence", FindingSeverity.BLOCKING, "The CR targets PROD (explicitly or by workflow context) but supplied evidence indicates only DEV validation.", recommendation="Provide production-equivalent non-PROD validation evidence and reconcile the environment claim."))

    if any(f.severity == FindingSeverity.BLOCKING for f in findings):
        decision = Decision.NOT_READY
    elif any(f.severity == FindingSeverity.WARNING for f in findings):
        decision = Decision.CONDITIONAL
    else:
        decision = Decision.PASS

    confidence = 0.97 if decision == Decision.PASS else 0.88 if decision == Decision.CONDITIONAL else 0.84
    verified["document_agent_summary"] = specialist
    return EvidenceResult(decision, confidence, findings, verified, contradictions, [d.ref for d in documents])
