"""Deterministic guardrails for final Pre-CAB decisions."""
from __future__ import annotations

from .requirements import infer_requirements
from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness, ValidationResult

STRICTNESS_PENALTIES = {
    Strictness.LENIENT: {"warning": 2},
    Strictness.BALANCED: {"warning": 5},
    Strictness.STRICT: {"warning": 10},
}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def classify_change(cr: dict) -> str:
    return _text(cr.get("Type")).lower()


def infer_uat_requirement(cr: dict) -> tuple[bool, str]:
    prediction = next((p for p in infer_requirements(cr) if p.name == "UAT"), None)
    return (prediction.required, prediction.reason) if prediction else (False, "No UAT requirement prediction was produced.")


def rollback_quality(backout_plan: object) -> tuple[bool, str]:
    text = _text(backout_plan).lower()
    if not text:
        return False, "No backout/recovery mechanism is described."
    if text in {"na", "n/a", "none", "not applicable"}:
        return False, "Backout is marked not applicable without supporting rationale."
    recovery_terms = (
        "backup", "restore", "rollback", "revert", "previous version", "snapshot",
        "restore services", "replace", "recover", "back up", "can be reverted", "can be reversed",
    )
    if any(term in text for term in recovery_terms):
        return True, "A credible recovery mechanism is described; procedural detail can affect confidence."
    return False, "No clear recovery mechanism was identified in the backout plan."


def _find_requirement(requirements: list[Requirement], name: str) -> Requirement | None:
    needle = name.lower()
    return next((r for r in requirements if r.name.lower() == needle), None)


def validate_fields(cr: dict, strictness: Strictness = Strictness.BALANCED) -> ValidationResult:
    """Validate a Normal CR before document evidence is independently verified."""
    findings: list[Finding] = []
    requirements: list[Requirement] = []
    score = 100.0

    if classify_change(cr) != "normal":
        findings.append(Finding(
            "OUT_OF_SCOPE", "Change type out of V1 scope", FindingSeverity.BLOCKING,
            "V1 validates Normal CRs; this request is not Normal.",
            technical_detail=f"Type={_text(cr.get('Type'))!r}",
            recommendation="Route the CR through the appropriate change process.",
        ))

    required = {
        "Number": "Change identifier",
        "Short description": "Change summary",
        "Description": "Change description",
        "Justification": "Business/operational reason",
        "Implementation plan": "Implementation approach",
        "Backout plan": "Recovery path",
        "Test plan": "Testing approach or rationale",
    }
    for field, label in required.items():
        if not _text(cr.get(field)):
            severity = FindingSeverity.BLOCKING if strictness == Strictness.STRICT or field in {"Implementation plan", "Backout plan"} else FindingSeverity.WARNING
            findings.append(Finding(
                f"MISSING_{field.upper().replace(' ', '_')}", f"Missing {label}", severity,
                f"{label} is not populated in the CR.",
                technical_detail=f"Field {field!r} is empty.",
                recommendation=f"Provide {label.lower()} before CAB review.",
            ))
            score -= STRICTNESS_PENALTIES[strictness]["warning"]

    rollback_ok, rollback_reason = rollback_quality(cr.get("Backout plan"))
    findings.append(Finding(
        "BACKOUT_OK" if rollback_ok else "BACKOUT_WEAK",
        "Recovery path identified" if rollback_ok else "Recovery path needs attention",
        FindingSeverity.INFO if rollback_ok else (FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING),
        "The CR contains a plausible recovery mechanism." if rollback_ok else "The backout plan does not yet provide a clear recovery mechanism.",
        technical_detail=rollback_reason,
        recommendation="Add or justify a usable rollback/recovery path." if not rollback_ok else "",
    ))
    if not rollback_ok:
        score -= STRICTNESS_PENALTIES[strictness]["warning"]

    for prediction in infer_requirements(cr):
        requirements.append(Requirement(prediction.name, prediction.required, prediction.reason, source="context"))

    uat_req = _find_requirement(requirements, "UAT")
    if uat_req and uat_req.required:
        findings.append(Finding(
            "UAT_CONTEXT", "UAT is expected for this change", FindingSeverity.INFO,
            "Functional/customer-facing signals make UAT or equivalent validation applicable.",
            technical_detail=uat_req.reason,
        ))
        if not _text(cr.get("Test plan")):
            findings.append(Finding(
                "UAT_PLAN_MISSING", "UAT/testing plan missing", FindingSeverity.WARNING,
                "The change context suggests functional testing, but no testing plan is present.",
                recommendation="Provide the applicable functional/UAT test approach.",
            ))
    else:
        findings.append(Finding(
            "UAT_NOT_MANDATORY", "UAT not assumed mandatory", FindingSeverity.INFO,
            "The validator does not require UAT solely because the field exists.",
            technical_detail=uat_req.reason if uat_req else "No UAT prediction was produced.",
        ))

    customer_req = _find_requirement(requirements, "Customer approval")
    if customer_req and customer_req.required:
        approval = _text(cr.get("Customer Approval"))
        completed = approval.lower() in {"yes", "approved"}
        if completed:
            findings.append(Finding(
                "CUSTOMER_APPROVAL_PRESENT", "Customer approval recorded", FindingSeverity.INFO,
                "The CR records a completed customer approval state; attachment evidence is verified separately.",
                technical_detail=f"Customer Approval={approval!r}",
            ))
        else:
            findings.append(Finding(
                "CUSTOMER_APPROVAL_GAP", "Customer approval is expected", FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING,
                "The change context suggests customer approval should be verified, but the CR does not show a completed approval.",
                technical_detail=f"Customer Approval={approval!r}; reason={customer_req.reason}",
                recommendation="Obtain and attach the applicable customer approval evidence, or document an approved exception.",
            ))
            score -= STRICTNESS_PENALTIES[strictness]["warning"]

    conflict = _text(cr.get("Conflict status")).lower()
    if conflict == "conflict":
        findings.append(Finding(
            "CONFLICT", "Blocking change conflict", FindingSeverity.BLOCKING,
            "ServiceNow reports a conflict for this change.",
            technical_detail="Conflict status=Conflict",
            recommendation="Resolve or explicitly disposition the conflict before approval.",
        ))
    elif conflict in {"not run", ""}:
        findings.append(Finding(
            "CONFLICT_UNVERIFIED", "Conflict status is not fully verified", FindingSeverity.WARNING,
            "A current conflict check is not confirmed.",
            technical_detail=f"Conflict status={conflict!r}",
            recommendation="Run or verify the ServiceNow conflict check.",
        ))
        score -= STRICTNESS_PENALTIES[strictness]["warning"]
    else:
        findings.append(Finding(
            "NO_CONFLICT", "No blocking conflict reported", FindingSeverity.INFO,
            "ServiceNow reports no conflict for this change.",
            technical_detail=f"Conflict status={conflict!r}",
        ))

    if not _text(cr.get("Configuration item")):
        findings.append(Finding(
            "CI_MISSING", "Configuration item is not populated",
            FindingSeverity.WARNING if strictness != Strictness.STRICT else FindingSeverity.BLOCKING,
            "The affected production configuration item is not identified in the CR.",
            recommendation="Relate the applicable configuration item(s).",
        ))
        score -= STRICTNESS_PENALTIES[strictness]["warning"]

    blocking = [f for f in findings if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)
    score = max(0.0, min(100.0, score))
    confidence = max(0.50, min(0.99, 0.75 + (score / 100.0) * 0.24 - len(warnings) * 0.02))
    return ValidationResult(
        decision=decision, confidence=confidence, score=score, strictness=strictness,
        findings=findings, requirements=requirements,
        technical_summary="Field-level validation completed; attachments have not yet been independently verified.",
        cab_summary=decision.value.replace("_", " ").title(),
    )
