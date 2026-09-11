"""Deterministic guardrails for final Pre-CAB decisions."""
from __future__ import annotations

from .field_requirement_engine import FieldRequirementEngine, RequirementLevel
from .input_loader import normalize_change_type
from .requirements import infer_requirements
from .rules import context_flags, field_quality, required_field_policies
from .schemas import Decision, Finding, FindingSeverity, Requirement, Strictness, ValidationResult

STRICTNESS_PENALTIES = {
    Strictness.LENIENT: {"warning": 2},
    Strictness.BALANCED: {"warning": 5},
    Strictness.STRICT: {"warning": 10},
}

# Shared engine instance: loads config/field_requirements.generated.json once.
# Its findings are CANDIDATE-status (mined, not yet benchmarked/promoted) --
# see field_requirement_engine.py -- so they are surfaced here as evidence
# *alongside* the keyword-based field policy engine, capped at WARNING, never
# escalated to a hard BLOCKING gate on their own until someone promotes the
# rule table to VALIDATED/ACTIVE.
_FIELD_REQUIREMENT_ENGINE = FieldRequirementEngine()


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def classify_change(cr: dict) -> str:
    return normalize_change_type(cr.get("Type") or cr.get("change_type") or cr.get("Change Type") or cr.get("Change class") or "")


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


def _missing_severity(policy_name: str, declared: str, strictness: Strictness) -> FindingSeverity:
    """Apply governance severity while keeping strictness a tolerance control.

    Identity, implementation and recovery gaps are hard blockers. Other gaps are warnings in
    balanced/lenient mode and become blockers under strict mode when the rule declares them blocking.
    """
    if declared.upper() == "BLOCKING":
        return FindingSeverity.BLOCKING
    if strictness == Strictness.STRICT:
        return FindingSeverity.BLOCKING
    return FindingSeverity.WARNING


def validate_fields(cr: dict, strictness: Strictness = Strictness.BALANCED) -> ValidationResult:
    """Run the centralized Normal-CR field/rule engine before document evidence verification."""
    findings: list[Finding] = []
    requirements: list[Requirement] = []
    score = 100.0

    change_type = classify_change(cr)
    if change_type != "normal":
        findings.append(Finding(
            "OUT_OF_SCOPE", "Change type out of V1 scope", FindingSeverity.BLOCKING,
            "V1 validates Normal CRs; this request is not Normal.",
            technical_detail=f"Type={_text(cr.get('Type'))!r}",
            recommendation="Route the CR through the appropriate change process.",
        ))

    policies = required_field_policies(cr)
    flags = context_flags(cr)
    for policy, required, matched_flags in policies:
        requirements.append(
            Requirement(
                policy.field,
                required,
                (
                    f"Baseline field: {policy.label}."
                    if policy.baseline
                    else f"Required when: {', '.join(matched_flags) or ', '.join(policy.required_when)}."
                ),
                source="field-policy-v2",
            )
        )
        if required and not _text(cr.get(policy.field)):
            severity = _missing_severity(policy.field, policy.missing_severity, strictness)
            findings.append(Finding(
                f"MISSING_{policy.field.upper().replace(' ', '_').replace('/', '_')}",
                f"Missing {policy.label}",
                severity,
                f"{policy.label} is required for this CR under the active field policy.",
                technical_detail=f"Matched conditions={matched_flags or ('baseline',)}; domain={policy.domain}.",
                recommendation=f"Populate {policy.label.lower()} before CAB review.",
            ))
            score -= STRICTNESS_PENALTIES[strictness]["warning"]

    # Preserve the dedicated recovery semantic check: presence is not enough.
    rollback_ok, rollback_reason = rollback_quality(cr.get("Backout plan"))
    findings.append(Finding(
        "BACKOUT_OK" if rollback_ok else "BACKOUT_WEAK",
        "Recovery path identified" if rollback_ok else "Recovery path needs attention",
        FindingSeverity.INFO if rollback_ok else FindingSeverity.BLOCKING,
        "The CR contains a plausible recovery mechanism." if rollback_ok else "The backout plan does not yet provide a clear recovery mechanism.",
        technical_detail=rollback_reason,
        recommendation="Add or justify a usable rollback/recovery path." if not rollback_ok else "",
    ))
    if not rollback_ok:
        score -= STRICTNESS_PENALTIES[strictness]["warning"]

    # Contextual requirements remain visible even when their field is not populated.
    for prediction in infer_requirements(cr):
        if prediction.name not in {req.name for req in requirements}:
            requirements.append(Requirement(
                prediction.name, prediction.required, prediction.reason, source="contextual-rule-v2"
            ))

    uat_req = _find_requirement(requirements, "UAT")
    if uat_req and uat_req.required:
        findings.append(Finding(
            "UAT_CONTEXT", "UAT is expected for this change", FindingSeverity.INFO,
            "Functional/customer-facing signals make UAT or equivalent validation applicable.",
            technical_detail=uat_req.reason,
        ))
    else:
        findings.append(Finding(
            "UAT_NOT_MANDATORY", "UAT not assumed mandatory", FindingSeverity.INFO,
            "The validator does not require UAT solely because the field exists; applicability is contextual.",
            technical_detail=uat_req.reason if uat_req else "No UAT prediction was produced.",
        ))

    customer_req = _find_requirement(requirements, "Customer Approval")
    if customer_req and customer_req.required:
        approval = _text(cr.get("Customer Approval")).lower()
        completed = approval in {"yes", "approved", "approved by customer", "complete", "completed"}
        if completed:
            findings.append(Finding(
                "CUSTOMER_APPROVAL_PRESENT", "Customer approval recorded", FindingSeverity.INFO,
                "The CR records a completed customer approval state; document evidence is verified separately when available.",
                technical_detail=f"Customer Approval={approval!r}",
            ))
        else:
            severity = FindingSeverity.BLOCKING if strictness == Strictness.STRICT else FindingSeverity.WARNING
            findings.append(Finding(
                "CUSTOMER_APPROVAL_GAP", "Customer approval is expected", severity,
                "The change context suggests customer approval should be verified, but a completed approval is not recorded.",
                technical_detail=f"Customer Approval={approval!r}; reason={customer_req.reason}",
                recommendation="Obtain applicable customer approval evidence or document an approved exception.",
            ))
            score -= STRICTNESS_PENALTIES[strictness]["warning"]

    conflict = _text(cr.get("Conflict status")).lower()
    if conflict in {"conflict", "conflicted"}:
        findings.append(Finding(
            "CONFLICT", "Blocking change conflict", FindingSeverity.BLOCKING,
            "ServiceNow reports a conflict for this change.",
            technical_detail=f"Conflict status={conflict!r}",
            recommendation="Resolve or explicitly disposition the conflict before approval.",
        ))
    elif conflict in {"not run", "not checked", "unknown", ""}:
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
            "ServiceNow reports no blocking conflict for this change.",
            technical_detail=f"Conflict status={conflict!r}",
        ))

    # Surface quality scores as non-blocking intelligence so GPT-OSS can reason on quality rather
    # than treating every populated field as equally good.
    quality = field_quality(cr)
    low_quality = [item for item in quality if item.present and item.score < 0.45]
    if low_quality:
        findings.append(Finding(
            "FIELD_QUALITY_WEAK",
            "One or more populated fields are weak or placeholder-like",
            FindingSeverity.WARNING if strictness != Strictness.STRICT else FindingSeverity.BLOCKING,
            "Some required/contextual fields are present but contain weak or placeholder-like content.",
            technical_detail="; ".join(f"{item.field}={item.score:.2f}" for item in low_quality),
            recommendation="Replace placeholders with specific, testable operational detail.",
        ))
        score -= STRICTNESS_PENALTIES[strictness]["warning"]

    # Data-driven layer: what this org's own historical Normal CRs (by Category/Sub
    # Category) actually required, mined from real fill/disposition rates rather
    # than assumed from keywords (see field_requirement_engine.py). CANDIDATE-status
    # -- surfaced as evidence, capped at WARNING, never a hard gate on its own, and
    # skipped for a field already flagged by the keyword-policy checks above so CAB
    # doesn't see the same gap reported twice.
    already_flagged_fields = {
        code.removeprefix("MISSING_").replace("_", " ") for code in (f.code for f in findings) if code.startswith("MISSING_")
    } | {"Customer Approval"}  # CUSTOMER_APPROVAL_GAP/PRESENT above already covers this field
    fr_report = _FIELD_REQUIREMENT_ENGINE.evaluate(cr)
    for item in fr_report.findings:
        if item.satisfied or item.requirement_level not in (RequirementLevel.REQUIRED, RequirementLevel.CONDITIONAL):
            continue
        if item.field.upper() in {f.upper() for f in already_flagged_fields}:
            continue
        findings.append(Finding(
            f"HISTORICAL_GAP_{item.field.upper().replace(' ', '_').replace('/', '_')}",
            f"{item.field} gap vs. historical pattern",
            FindingSeverity.WARNING,
            item.rationale,
            technical_detail=f"{item.evidence} (scope={item.rule_scope}, confidence={item.confidence:.2f})",
            recommendation=f"Review whether {item.field.lower()} should be populated for this change.",
        ))
        score -= STRICTNESS_PENALTIES[strictness]["warning"] * 0.5  # softer weight: unvalidated evidence

    for note in fr_report.note_signals:
        findings.append(Finding(
            "HISTORICAL_NOTE_SIGNAL",
            "Work notes echo language seen before past rejections/cancellations",
            FindingSeverity.WARNING if note.severity == FindingSeverity.WARNING else FindingSeverity.INFO,
            f"This CR's own work notes contain the phrase {note.phrase!r}.",
            technical_detail=note.evidence,
            recommendation="Confirm this CR is not headed toward the same outcome as similar past CRs.",
        ))

    blocking = [f for f in findings if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)
    score = max(0.0, min(100.0, score))
    confidence = max(0.50, min(0.99, 0.72 + (score / 100.0) * 0.26 - len(warnings) * 0.015))

    return ValidationResult(
        decision=decision,
        confidence=confidence,
        score=score,
        strictness=strictness,
        findings=findings,
        requirements=requirements,
        technical_summary=(
            "Centralized field-policy validation completed; contextual requirements and field quality "
            "were evaluated, while attachments remain a separate evidence stage."
        ),
        cab_summary=decision.value.replace("_", " ").title(),
        metadata={
            "rule_engine_version": "2.0",
            "context_flags": flags,
            "required_field_count": sum(1 for _, required, _ in policies if required),
            "field_quality": [
                {"field": item.field, "present": item.present, "score": item.score, "reasons": list(item.reasons)}
                for item in quality
            ],
            "field_requirement_engine": {
                "lifecycle_status": _FIELD_REQUIREMENT_ENGINE.lifecycle_status,
                "version": _FIELD_REQUIREMENT_ENGINE.version,
                "resolved_scope": fr_report.resolved_scope,
            },
        },
    )
