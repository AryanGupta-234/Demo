"""CAB-facing narrative rendering.

This module only formats evidence already computed by the validator. It does not
re-decide the CR, invent evidence, or treat the CAB outcome fields as inputs to
the prediction.
"""
from __future__ import annotations

from textwrap import shorten
from typing import Any

from .schemas import Decision, FindingSeverity

_NOT_APPLICABLE_VALUES = {"", "na", "n/a", "none", "not applicable"}

_CHECKLIST_REPRESENTED_CODES = {
    "MISSING_CONFIGURATION_ITEM", "MISSING_IMPLEMENTATION_PLAN", "BACKOUT_OK", "BACKOUT_WEAK",
    "MISSING_TEST_PLAN", "MISSING_CUSTOMER_APPROVAL", "CUSTOMER_APPROVAL_GAP", "CUSTOMER_APPROVAL_PRESENT",
    "UAT_CONTEXT", "UAT_NOT_MANDATORY",
}

_CHAIN_AGENTS = [
    ("technical", "Technical Agent"),
    ("testing", "Testing Agent"),
    ("risk", "Risk Agent"),
    ("clone", "Clone Agent"),
]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_meaningful(value: Any) -> bool:
    return _text(value).lower() not in _NOT_APPLICABLE_VALUES


def _excerpt(value: Any, width: int = 180) -> str:
    text = " ".join(_text(value).split())
    return shorten(text, width=width, placeholder=" …") if text else "Not provided"


def format_agent_chains(agent_results: list[Any]) -> str:
    """Render the specialist-agent reasoning chains as arrow-bullet blocks."""
    by_name = {result.agent: result for result in agent_results}
    blocks: list[str] = []
    for key, label in _CHAIN_AGENTS:
        result = by_name.get(key)
        chain = result.notes.get("chain") if result else None
        if not chain:
            continue
        lines = [label] + [f"→ {line}" for line in chain]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _checklist(
    cr: dict[str, Any], agent_results: list[Any], top_clone: dict[str, Any] | None, findings: list[Any]
) -> list[str]:
    by_name = {result.agent: result for result in agent_results}
    technical = by_name.get("technical")
    testing = by_name.get("testing")
    items: list[tuple[bool, str]] = []

    category = _text(cr.get("Category"))
    items.append((bool(category), f"Application/change category identified ({category})" if category else "Application/change category not identified"))

    ci = _text(cr.get("Configuration item"))
    environment = _text(cr.get("Environment")).lower()
    items.append((bool(ci), f"Configuration item identified{' (Prod)' if 'prod' in environment else ''}" if ci else "No configuration item identified"))

    implementation = _is_meaningful(cr.get("Implementation plan"))
    items.append((implementation, "Implementation steps present" if implementation else "Implementation steps missing"))

    rollback_aligned = bool(technical and technical.notes.get("rollback_aligned"))
    items.append((rollback_aligned, "Recovery/rollback path identified" if rollback_aligned else "Recovery/rollback path not confirmed"))

    profile = (by_name.get("context").notes.get("change_profile") if by_name.get("context") else None) or {}
    non_prod_expected = bool(profile.get("non_prod_validation_expected"))
    pre_prod = bool(testing and testing.notes.get("pre_prod_claimed"))
    if non_prod_expected:
        items.append((pre_prod, "Non-PROD validation stated" if pre_prod else "Non-PROD validation not confirmed"))
    else:
        items.append((True, "Non-PROD validation not mandatory for this change context"))

    test_plan = _is_meaningful(cr.get("Test plan"))
    items.append((test_plan, "Test plan present" if test_plan else "Test plan missing"))

    customer_approval = _text(cr.get("Customer Approval"))
    if customer_approval:
        approval_ok = customer_approval.lower() in {"yes", "approved", "complete", "completed"}
        items.append((approval_ok, f"Customer approval = {customer_approval}"))

    if top_clone and top_clone.get("historical_decision"):
        historical_ok = str(top_clone["historical_decision"]).strip().lower() == "approved"
        items.append((historical_ok, f"Historical CAB pattern = {top_clone['historical_decision']}"))

    lines = [f"{_mark(ok)} {label}" for ok, label in items]
    seen_codes: set[str] = set()
    for finding in findings:
        code = getattr(finding, "code", "")
        severity = getattr(finding, "severity", None)
        if code in _CHECKLIST_REPRESENTED_CODES or code in seen_codes:
            continue
        if severity == FindingSeverity.BLOCKING:
            lines.append(f"❌ {getattr(finding, 'title', code)}")
            seen_codes.add(code)
        elif severity == FindingSeverity.WARNING:
            lines.append(f"⚠️ {getattr(finding, 'title', code)}")
            seen_codes.add(code)
    return lines


def _key_gaps(findings: list[Any], limit: int = 6) -> list[Any]:
    # Decision drivers are material readiness items only. Informational observations
    # remain visible in detailed sections but never crowd the actual reason.
    material = [
        f for f in findings
        if getattr(f, "severity", FindingSeverity.INFO) in {FindingSeverity.BLOCKING, FindingSeverity.WARNING}
    ]
    ranked = {FindingSeverity.BLOCKING: 0, FindingSeverity.WARNING: 1}
    return sorted(material, key=lambda f: ranked.get(getattr(f, "severity", FindingSeverity.WARNING), 1))[:limit]


def _decision_summary(
    cr: dict[str, Any],
    decision: Decision,
    findings: list[Any],
    brain_payload: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Build a CAB-readable decision paragraph plus explicit decision drivers.

    Decision drivers are computed once, from the deterministic findings,
    regardless of whether the model ran — a CAB reviewer (technical or not)
    needs the "why" list either way, and previously the AI-summary branch
    below returned an empty driver list every time the model actually ran,
    which meant every AI-assisted report showed "See evidence and findings
    below." for its Decision drivers, useful info discarded for nothing.
    """
    blocking = [f for f in findings if getattr(f, "severity", None) == FindingSeverity.BLOCKING]
    warnings = [f for f in findings if getattr(f, "severity", None) == FindingSeverity.WARNING]
    gaps = _key_gaps(findings)
    names = [getattr(f, "title", "validation gap") for f in gaps if getattr(f, "code", "")]

    if brain_payload:
        cab_reasoning = _text(brain_payload.get("cab_reasoning"))
        technical_reasoning = _text(brain_payload.get("technical_reasoning"))
        if cab_reasoning:
            parts = [cab_reasoning]
            if technical_reasoning:
                parts.append(f"Technical basis: {technical_reasoning}")
            return " ".join(parts), names[:5]

    if decision == Decision.PASS:
        summary = (
            "The CR meets the currently evaluated readiness checks. The available record contains "
            "sufficient implementation, testing, risk, and approval evidence for a CAB-ready assessment."
        )
    elif decision == Decision.CONDITIONAL:
        summary = (
            "The CR is not cleanly CAB-ready yet. Core change information is present, but one or more "
            "review gaps remain that should be resolved or explicitly accepted before CAB approval."
        )
    else:
        summary = (
            "The CR is not CAB-ready in its current form. The validator identified material evidence or "
            "readiness gaps that prevent the change from being supported by a sufficiently complete "
            "technical, testing, risk, or operational record."
        )
    if blocking:
        summary += f" {len(blocking)} blocking issue(s) were identified."
    elif warnings:
        summary += f" {len(warnings)} warning-level issue(s) remain."
    return summary, names[:5]


def _evidence_snapshot(cr: dict[str, Any]) -> list[str]:
    rows = [
        ("Short description", cr.get("Short description")),
        ("Description", cr.get("Description")),
        ("Justification", cr.get("Justification")),
        ("Implementation plan", cr.get("Implementation plan")),
        ("Change plan", cr.get("Change plan")),
        ("Backout plan", cr.get("Backout plan")),
        ("Work notes", cr.get("Work notes")),
        ("Comments", cr.get("Comments")),
        ("Test plan", cr.get("Test plan")),
        ("Configuration item", cr.get("Configuration item")),
        ("Risk", cr.get("Risk")),
        ("Priority", cr.get("Priority")),
    ]
    return [
        f"{label:<21} {'PRESENT' if _is_meaningful(value) else 'MISSING/UNSET':<12} {_excerpt(value)}"
        for label, value in rows
    ]


def _signoff_snapshot(cr: dict[str, Any]) -> list[str]:
    fields = (
        "UAT signoff", "Customer Approval", "TCS QA signoff",
        "Test Results Evidence", "Lower Environment Reference CR/SR",
    )
    return [f"{field:<33} {_text(cr.get(field)) or 'Not recorded'}" for field in fields]


def _agent_detail(agent_results: list[Any]) -> list[str]:
    lines: list[str] = []
    for result in agent_results:
        label = result.agent.replace("_", " ").title()
        notes = result.notes or {}
        chain = notes.get("chain") or []
        if not chain:
            continue
        lines.append(label)
        for item in chain:
            lines.append(f"  → {item}")
        if notes.get("rollback_reason"):
            lines.append(f"  Evidence note: {notes['rollback_reason']}")
        if notes.get("technical_uncertainty"):
            lines.append(f"  Uncertainty: {notes['technical_uncertainty'].upper()}")
        if notes.get("risk_confidence") is not None:
            lines.append(f"  Risk assessment confidence: {float(notes['risk_confidence']):.0%}")
        lines.append("")
    return lines[:-1] if lines else []


def _cab_questions(cr: dict[str, Any], agent_results: list[Any], brain_payload: dict[str, Any] | None) -> list[str]:
    if brain_payload and isinstance(brain_payload.get("cab_questions"), list) and brain_payload["cab_questions"]:
        return [str(q) for q in brain_payload["cab_questions"][:6]]

    by_name = {result.agent: result for result in agent_results}
    technical = by_name.get("technical")
    testing = by_name.get("testing")
    risk = by_name.get("risk")
    questions: list[str] = []
    context_result = by_name.get("context")
    profile = (context_result.notes.get("change_profile") if context_result else None) or {}
    if not _text(cr.get("Configuration item")) and profile.get("primary_archetype") in {"FUNCTIONAL", "DATABASE", "NETWORK"}:
        questions.append("Which production Configuration Item(s) are actually being changed?")
    if testing and testing.notes.get("functional_coverage_ok") is False:
        questions.append("What validation is still needed to demonstrate the intended outcome?")
    elif testing and profile.get("formal_test_evidence_expected") and not testing.notes.get("execution_claimed") and not testing.notes.get("formal_evidence_present"):
        questions.append("When the planned validation is executed, where will the result be recorded?")
    if risk and not _text(cr.get("Risk")) and (profile.get("high_impact") or profile.get("customer_facing") or profile.get("security")):
        questions.append("What is the formal risk classification and who assessed it?")
    if technical and not technical.notes.get("rollback_aligned"):
        questions.append("What is the exact rollback trigger, procedure, and recovery validation?")
    if technical and technical.notes.get("dependency_evidence_present") is False and profile.get("high_impact"):
        questions.append("Is there any dependency or sequencing detail that the implementation team must coordinate?")
    return questions[:6] or ["Are there any residual operational risks or dependencies not captured in the CR?"]


def _recommendations(cr: dict[str, Any], findings: list[Any], brain_payload: dict[str, Any] | None = None) -> list[str]:
    deterministic: list[str] = []
    for finding in findings:
        severity = getattr(finding, "severity", None)
        if severity not in {FindingSeverity.BLOCKING, FindingSeverity.WARNING}:
            continue
        rec = getattr(finding, "recommendation", "") or ""
        if rec and rec not in deterministic:
            deterministic.append(rec)
    deterministic = deterministic[:4]

    model_recs: list[str] = []
    if brain_payload and isinstance(brain_payload.get("recommendations"), list):
        for rec in brain_payload["recommendations"]:
            text = str(rec)
            if text and text not in deterministic and text not in model_recs:
                model_recs.append(text)
    model_recs = model_recs[:3]

    recommendations = deterministic + model_recs
    if not recommendations:
        recommendations.append("Confirm final CAB evidence before approval.")
    return recommendations


def format_cab_result(
    cr_number: str,
    cr: dict[str, Any],
    decision: Decision,
    confidence: float,
    agent_results: list[Any],
    findings: list[Any],
    brain_payload: dict[str, Any] | None = None,
) -> str:
    """Render a detailed CAB-facing validation report."""
    by_name = {result.agent: result for result in agent_results}
    clone_agent = by_name.get("clone")
    clone_candidates = clone_agent.notes.get("clone_candidates", []) if clone_agent else []
    top_clone = clone_candidates[0] if clone_candidates else None
    summary, drivers = _decision_summary(cr, decision, findings, brain_payload)
    has_observations = any(
        getattr(f, "severity", FindingSeverity.INFO) == FindingSeverity.INFO
        and getattr(f, "code", "") not in {"CHANGE_TYPE_CONTEXT", "TARGET_ENVIRONMENT_DERIVED", "UAT_NOT_MANDATORY"}
        for f in findings
    )
    prediction_label = {
        Decision.PASS: "✅ CAB-READY WITH OBSERVATIONS" if has_observations else "✅ CAB-READY",
        Decision.CONDITIONAL: "⚠️ CONDITIONAL — REVIEW REQUIRED",
        Decision.NOT_READY: "❌ NOT READY",
    }[decision]

    lines = [
        "PRE-CAB VALIDATION REPORT",
        "═" * 68,
        "",
        f"CR: {cr_number}",
        f"Decision: {prediction_label}",
        f"Confidence: {confidence:.0%}",
        "Validation mode: Balanced",
        "Decision basis: Deterministic policy + specialist reasoning",
        "",
        "─" * 68,
        "1. EXECUTIVE SUMMARY",
        "─" * 68,
        summary,
        "",
        "Decision drivers:",
    ]
    for driver in drivers or ["See evidence and findings below."]:
        lines.append(f"  • {driver}")

    lines.extend([
        "",
        "What this means for CAB:",
        f"  {_cab_cab_implication(decision)}",
        "",
        "─" * 68,
        "2. EVIDENCE REVIEWED",
        "─" * 68,
        "Core CR fields:",
    ])
    lines.extend([f"  {line}" for line in _evidence_snapshot(cr)])
    lines.append("")
    lines.append("Signoff / evidence dispositions:")
    lines.extend([f"  {line}" for line in _signoff_snapshot(cr)])

    lines.extend([
        "",
        "─" * 68,
        "3. TECHNICAL ASSESSMENT",
        "─" * 68,
    ])
    technical = by_name.get("technical")
    if technical:
        n = technical.notes
        lines.extend([
            f"Implementation: {_mark(bool(_is_meaningful(cr.get('Implementation plan'))))}",
            f"  Evidence: {_excerpt(cr.get('Implementation plan'), 260)}",
            f"Configuration item: {'✅ Identified' if _text(cr.get('Configuration item')) else '⚠️ Not identified (traceability observation)'}",
            f"  Value: {_text(cr.get('Configuration item')) or 'Not identified'}",
            f"Target environment: {n.get('effective_environment') or ('PROD' if not _text(cr.get('Environment')) else _text(cr.get('Environment')))}",
            f"Rollback/recovery: {_mark(bool(n.get('rollback_aligned')))}",
            f"  Assessment: {_excerpt(n.get('rollback_reason') or cr.get('Backout plan'), 260)}",
            f"Non-PROD validation context: {_mark(bool(n.get('non_prod_validation_claimed')))}",
            "  SIT/UAT/Pre-PROD/lower environment are treated as one validation class",
            f"Dependency/sequencing context: {_mark(bool(n.get('dependency_evidence_present')))}",
            f"Technical uncertainty: {_text(n.get('technical_uncertainty')).upper() or 'UNKNOWN'}",
        ])
    else:
        lines.append("Technical agent output was not available.")

    lines.extend([
        "",
        "─" * 68,
        "4. TESTING & VALIDATION ASSESSMENT",
        "─" * 68,
    ])
    testing = by_name.get("testing")
    if testing:
        n = testing.notes
        lines.extend([
            f"Test plan: {_mark(_is_meaningful(cr.get('Test plan')))}",
            f"  Evidence: {_excerpt(cr.get('Test plan'), 260)}",
            f"Change profile: {((by_name.get('context').notes.get('change_profile') if by_name.get('context') else {}) or {}).get('primary_archetype', 'GENERAL')}",
            f"Formal test-results evidence applicable: {_mark(bool(((by_name.get('context').notes.get('change_profile') if by_name.get('context') else {}) or {}).get('formal_test_evidence_expected')))}",
            f"Non-PROD validation expected: {_mark(bool(((by_name.get('context').notes.get('change_profile') if by_name.get('context') else {}) or {}).get('non_prod_validation_expected')))}",
            f"Test execution result in Work Notes/Comments: {_mark(bool(n.get('execution_claimed')))}",
            f"  Journal evidence: {_excerpt(' '.join(str(cr.get(k) or '') for k in ('Work notes', 'Comments')), 220)}",
            f"Formal Test Results Evidence: {_mark(bool(n.get('formal_evidence_present')))}",
            f"  Evidence field: {_excerpt(cr.get('Test Results Evidence'), 220)}",
            f"Functional/technical validation coverage: {_mark(bool(n.get('functional_coverage_ok')))}",
        ])
    else:
        lines.append("Testing agent output was not available.")
    uat_line, uat_reason = _uat_summary(findings)
    lines.append(f"UAT: {uat_line}")
    if uat_reason:
        lines.append(f"  Basis: {_excerpt(uat_reason, 300)}")

    lines.extend([
        "",
        "─" * 68,
        "5. RISK & OPERATIONAL IMPACT",
        "─" * 68,
    ])
    risk = by_name.get("risk")
    risk_value = _text(cr.get("Risk")) or "unknown"
    lines.append(f"Declared risk: {risk_value}")
    if risk:
        risk_notes = risk.notes
        lines.append(f"Impact consistency: {'CONTRADICTION' if risk_notes.get('contradiction') else 'NO DIRECT CONTRADICTION'}")
        lines.append(f"Risk confidence: {float(risk_notes.get('risk_confidence', 0.0)):.0%}")
    impact = cr.get("Risk and impact analysis")
    lines.append(f"Impact narrative: {_excerpt(impact, 380)}")
    lines.append("Interpretation: an unknown or weakly declared risk level is not evidence of zero risk.")

    lines.extend([
        "",
        "─" * 68,
        "6. HISTORICAL / CLONE ANALYSIS",
        "─" * 68,
    ])
    if top_clone:
        lines.append(f"Similar historical changes: {len(clone_candidates)}")
        lines.append(f"Strongest match: {top_clone.get('change_id', 'Unknown')} ({float(top_clone.get('similarity', 0.0)):.0%} similarity)")
        if top_clone.get("historical_decision"):
            lines.append(f"Historical outcome: {top_clone['historical_decision']}")
        if top_clone.get("changed_fields"):
            lines.append(f"Fields requiring revalidation: {', '.join(top_clone['changed_fields'][:6])}")
        lines.append(f"Reuse guidance: {top_clone.get('recommendation', 'Revalidate all change-specific evidence.')}")
    else:
        lines.append("No sufficiently similar historical change was identified.")
        lines.append("Therefore historical precedent does not independently support or reject this CR.")

    lines.extend([
        "",
        "─" * 68,
        "7. DECISION TRACE",
        "─" * 68,
        "Positive evidence established:",
    ])
    positives = [
        (bool(_is_meaningful(cr.get("Implementation plan"))), "Implementation procedure is populated"),
        (bool(_is_meaningful(cr.get("Backout plan"))), "Backout/recovery information is populated"),
        (bool(_is_meaningful(cr.get("Test plan"))), "Test plan is populated"),
        (bool(_text(cr.get("Risk"))), "Formal risk value is present"),
        (bool(by_name.get("testing") and by_name["testing"].notes.get("functional_coverage_ok")), "Applicable validation coverage is defined"),
    ]
    for ok, label in positives:
        lines.append(f"  {_mark(ok)} {label}")

    lines.append("")
    lines.append("Material unresolved issues:")
    material = [f for f in findings if getattr(f, "severity", None) in {FindingSeverity.BLOCKING, FindingSeverity.WARNING}]
    if material:
        for finding in material[:8]:
            lines.append(f"  {'❌' if finding.severity == FindingSeverity.BLOCKING else '⚠️'} {finding.title}: {_excerpt(getattr(finding, 'message', ''), 240)}")
    else:
        lines.append("  None identified by deterministic validation.")

    lines.extend([
        "",
        f"Final decision: {prediction_label}",
        f"Why: {_decision_reason(decision, material)}",
        "",
        "─" * 68,
        "8. CAB QUESTIONS",
        "─" * 68,
    ])
    for index, question in enumerate(_cab_questions(cr, agent_results, brain_payload), 1):
        lines.append(f"{index}. {question}")

    lines.extend([
        "",
        "─" * 68,
        "9. REQUIRED ACTIONS BEFORE CAB",
        "─" * 68,
    ])
    recommendations = _recommendations(cr, findings, brain_payload)
    for index, rec in enumerate(recommendations, 1):
        lines.append(f"Priority {index}: {rec}")

    lines.extend([
        "",
        "─" * 68,
        "10. SPECIALIST REASONING TRACE",
        "─" * 68,
    ])
    lines.extend(_agent_detail(agent_results) or ["No specialist reasoning chain was available."])
    if brain_payload:
        for key, title in (("technical_reasoning", "Model technical reasoning"), ("cab_reasoning", "Model CAB reasoning"), ("uncertainties", "Model uncertainties"), ("contradictions", "Model contradictions"), ("self_critique", "Model self-critique")):
            value = brain_payload.get(key)
            if value:
                lines.extend(["", title + ":"])
                if isinstance(value, list):
                    lines.extend([f"  • {_text(item)}" for item in value[:6]])
                else:
                    lines.extend([f"  {_text(value)}"])

    lines.extend([
        "",
        "CAB recommendation:",
        f"  {_cab_cab_recommendation(decision, material)}",
    ])
    return "\n".join(lines)


def _cab_cab_implication(decision: Decision) -> str:
    if decision == Decision.PASS:
        return "The CR can proceed to CAB review as a readiness-cleared change, subject to normal CAB authority."
    if decision == Decision.CONDITIONAL:
        return "CAB should review the listed open items and confirm whether the residual gaps can be explicitly accepted or must be closed first."
    return "Do not treat the CR as CAB-ready until the material evidence gaps are resolved or formally dispositioned."


def _decision_reason(decision: Decision, material: list[Any]) -> str:
    if decision == Decision.PASS:
        return "Required evidence is sufficiently complete and no unresolved material validation issue is driving a non-pass decision."
    titles = [getattr(f, "title", "material gap") for f in material[:4]]
    reason = "The remaining evidence gaps materially reduce readiness confidence."
    if titles:
        reason += " Primary drivers: " + "; ".join(titles) + "."
    return reason


def _cab_cab_recommendation(decision: Decision, material: list[Any]) -> str:
    if decision == Decision.PASS:
        return "CAB review may proceed; validate any last-minute operational changes before implementation."
    if decision == Decision.CONDITIONAL:
        return "Proceed only after the listed conditions are explicitly closed, accepted, or documented by the responsible owner."
    return "Do not approve as CAB-ready until the material gaps above are resolved."


def _uat_summary(findings: list[Any]) -> tuple[str, str]:
    uat_context = next((f for f in findings if getattr(f, "code", "") == "UAT_CONTEXT"), None)
    uat_not_mandatory = next((f for f in findings if getattr(f, "code", "") == "UAT_NOT_MANDATORY"), None)
    if uat_context:
        return "⚠️ Required for this change.", uat_context.technical_detail
    if uat_not_mandatory:
        return "✅ Not mandatory for this change.", uat_not_mandatory.technical_detail
    return "No UAT determination available.", ""
