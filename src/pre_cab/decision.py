"""Deterministic guardrails for final Pre-CAB decisions."""
from __future__ import annotations

from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness, ValidationResult


STRICTNESS_PENALTIES = {
    Strictness.LENIENT: {"warning": 2, "blocking": 100},
    Strictness.BALANCED: {"warning": 5, "blocking": 100},
    Strictness.STRICT: {"warning": 10, "blocking": 100},
}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _is_no(value: object) -> bool:
    return _text(value).lower() in {"no", "false"}


def classify_change(cr: dict) -> str:
    return _text(cr.get("Type")).lower()


def infer_uat_requirement(cr: dict) -> tuple[bool, str]:
    """Infer UAT from context; it is not a universal field requirement."""
    if classify_change(cr) != "normal":
        return False, "V1 targets Normal CRs; UAT inference is scoped to Normal changes."

    category = _text(cr.get("Category")).lower()
    subcategory = _text(cr.get("Sub Category")).lower()
    description = " ".join(_text(cr.get(k)) for k in ("Short description", "Description", "Justification")).lower()

    infra_terms = ("patch", "patching", "os patch", "security update", "server reboot")
    functional_terms = (
        "enhancement", "defect", "fix", "transaction", "sms", "payment",
        "report", "workflow", "customer", "interface", "api", "functional",
    )

    if category in {"infrastructure", "infra"} and any(t in description for t in infra_terms):
        return False, "Infrastructure/security maintenance is generally validated through operational or sanity checks."
    if any(t in description for t in functional_terms):
        return True, "The CR indicates a functional or customer-facing application change."
    if subcategory in {"atm transactions", "branchchannel", "esb", "api", "credittransfer"}:
        return True, "The change context suggests application or transaction behaviour may change."
    return False, "No strong functional-change signal was found; UAT is not assumed mandatory."


def rollback_quality(backout_plan: object) -> tuple[bool, str]:
    text = _text(backout_plan).lower()
    if not text:
        return False, "No backout/recovery mechanism is described."
    if text in {"na", "n/a", "none", "not applicable"}:
        return False, "Backout is explicitly marked not applicable with no supporting rationale."

    recovery_terms = (
        "backup", "restore", "rollback", "revert", "previous version", "snapshot",
        "restore services", "replace", "recover", "back up",
    )
    if any(term in text for term in recovery_terms):
        return True, "A credible recovery mechanism is described; procedural detail can affect confidence."
    if "can be reverted" in text or "can be reversed" in text:
        return True, "The CR states that the change is reversible; the procedure is brief."
    return False, "No clear recovery mechanism was identified in the backout plan."


def validate_fields(cr: dict, strictness: Strictness = Strictness.BALANCED) -> ValidationResult:
    """First-stage field validation. This never claims that attachments are verified."""
    findings: list[Finding] = []
    requirements: list[Requirement] = []
    score = 100.0

    if classify_change(cr) != "normal":
        findings.append(
            Finding(
                code="OUT_OF_SCOPE",
                title="Change type out of V1 scope",
                severity=FindingSeverity.BLOCKING,
                message="V1 validates Normal CRs; this request is not Normal.",
                technical_detail=f"Type={_text(cr.get('Type'))!r}",
                recommendation="Route the CR through the appropriate change process.",
            )
        )

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
            findings.append(
                Finding(
                    code=f"MISSING_{field.upper().replace(' ', '_')}",
                    title=f"Missing {label}",
                    severity=severity,
                    message=f"{label} is not populated in the CR.",
                    technical_detail=f"Field {field!r} is empty.",
                    recommendation=f"Provide {label.lower()} before CAB review.",
                )
            )
            score -= STRICTNESS_PENALTIES[strictness]["warning"]

    rollback_ok, rollback_reason = rollback_quality(cr.get("Backout plan"))
    if rollback_ok:
        findings.append(Finding("BACKOUT_OK", "Recovery path identified", FindingSeverity.INFO,
            "The CR contains a plausible recovery mechanism.", technical_detail=rollback_reason))
    else:
        findings.append(Finding("BACKOUT_WEAK", "Recovery path needs attention",
            FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING,
            "The backout plan does not yet provide a clear recovery mechanism.",
            technical_detail=rollback_reason, recommendation="Add or justify a usable rollback/recovery path."))
        score -= STRICTNESS_PENALTIES[strictness]["warning"]

    uat_required, uat_reason = infer_uat_requirement(cr)
    requirements.append(Requirement("UAT", uat_required, uat_reason, source="context"))

    if uat_required:
        test = _text(cr.get("Test plan"))
        uat = _text(cr.get("UAT signoff"))
        if not test:
            findings.append(Finding("UAT_PLAN_MISSING", "UAT requirement inferred", FindingSeverity.WARNING,
                "The change context suggests functional testing, but no testing plan is present.",
                technical_detail=uat_reason, recommendation="Provide the applicable functional/UAT test approach."))
        elif _is_no(uat) and strictness == Strictness.STRICT:
            findings.append(Finding("UAT_SIGNOFF_NO", "UAT signoff is negative", FindingSeverity.BLOCKING,
                "The CR indicates UAT/signoff is not complete for a change where UAT is expected.",
                technical_detail=f"UAT signoff={uat!r}", recommendation="Resolve the UAT/signoff gap or document an approved exception."))
    else:
        findings.append(Finding("UAT_NOT_MANDATORY", "UAT not assumed mandatory", FindingSeverity.INFO,
            "The validator does not require UAT solely because the field exists.", technical_detail=uat_reason))

    conflict = _text(cr.get("Conflict status")).lower()
    if conflict == "conflict":
        findings.append(Finding("CONFLICT", "Blocking change conflict", FindingSeverity.BLOCKING,
            "ServiceNow reports a conflict for this change.", technical_detail="Conflict status=Conflict",
            recommendation="Resolve or explicitly disposition the conflict before approval."))
    elif conflict in {"not run", ""}:
        findings.append(Finding("CONFLICT_UNVERIFIED", "Conflict status is not fully verified", FindingSeverity.WARNING,
            "A current conflict check is not confirmed.", technical_detail=f"Conflict status={_text(cr.get('Conflict status'))!r}",
            recommendation="Run or verify the ServiceNow conflict check."))
    else:
        findings.append(Finding("NO_CONFLICT", "No blocking conflict reported", FindingSeverity.INFO,
            "ServiceNow reports no conflict for this change.", technical_detail=f"Conflict status={_text(cr.get('Conflict status'))!r}"))

    if not _text(cr.get("Configuration item")):
        findings.append(Finding("CI_MISSING", "Configuration item is not populated",
            FindingSeverity.WARNING if strictness != Strictness.STRICT else FindingSeverity.BLOCKING,
            "The affected production configuration item is not identified in the CR.",
            recommendation="Relate the applicable configuration item(s)."))

    blocking = [f for f in findings if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]
    score = max(0.0, min(100.0, score))
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)
    confidence = max(0.50, min(0.99, 0.75 + (score / 100.0) * 0.24 - len(warnings) * 0.02))
    return ValidationResult(
        decision=decision,
        confidence=confidence,
        score=score,
        strictness=strictness,
        findings=findings,
        requirements=requirements,
        technical_summary="Field-level validation completed; attachments have not yet been independently verified.",
        cab_summary=decision.value.replace("_", " ").title(),
    )
