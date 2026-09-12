"""CAB-facing narrative rendering.

Two output shapes, both derived strictly from data already computed elsewhere
in the pipeline (agent notes, findings, requirements, clone matches) -- this
module formats, it never re-decides or invents a signal that isn't backed by
an actual field or an actual finding:

1. format_agent_chains(): a terse "Agent Name / → line / → line" narrative per
   specialist agent, for the agents whose notes carry a `chain` (see agents.py).
2. format_cab_result(): the final checklist-style CAB summary block.

Notably absent from the checklist: a "Technical/CAB approval = Approved" line.
This org's real ServiceNow export has no distinct pre-CAB technical-approval
field -- the only "Approval"/"CAB recommendation" fields ARE the CAB's own
outcome, and using them as an input here would mean grading a CR against the
answer the tool is supposed to be predicting. Checklist items are limited to
fields that exist independently of the CAB's own decision.
"""
from __future__ import annotations

from typing import Any

from .schemas import Decision

_NOT_APPLICABLE_VALUES = {"", "na", "n/a", "none", "not applicable"}

# Agents rendered as a reasoning chain, in display order, with a display name.
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


def _checklist(cr: dict[str, Any], agent_results: list[Any], top_clone: dict[str, Any] | None) -> list[str]:
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
    items.append((rollback_aligned, "Backup/revert mechanism identified" if rollback_aligned else "Backup/revert mechanism not confirmed"))

    pre_prod = bool(testing and testing.notes.get("pre_prod_claimed"))
    items.append((pre_prod, "Pre-PROD testing stated" if pre_prod else "No pre-PROD testing stated"))

    test_plan = _is_meaningful(cr.get("Test plan"))
    items.append((test_plan, "Test plan present" if test_plan else "Test plan missing"))

    customer_approval = _text(cr.get("Customer Approval"))
    approval_ok = customer_approval.lower() in {"yes", "approved", "complete", "completed"}
    if customer_approval:
        items.append((approval_ok, f"Customer approval = {customer_approval}"))

    if top_clone and top_clone.get("historical_decision"):
        historical_ok = str(top_clone["historical_decision"]).strip().lower() == "approved"
        items.append((historical_ok, f"Historical CAB pattern = {top_clone['historical_decision']}"))

    return [f"{_mark(ok)} {label}" for ok, label in items]


def _cab_questions(cr: dict[str, Any], agent_results: list[Any], brain_payload: dict[str, Any] | None) -> list[str]:
    if brain_payload and isinstance(brain_payload.get("cab_questions"), list) and brain_payload["cab_questions"]:
        return [str(q) for q in brain_payload["cab_questions"][:4]]

    by_name = {result.agent: result for result in agent_results}
    testing = by_name.get("testing")
    risk = by_name.get("risk")
    questions: list[str] = []
    if testing and not testing.notes.get("functional_coverage_ok"):
        questions.append("What functional scenarios were tested?")
    if not testing or not testing.notes.get("evidence_present"):
        questions.append("What is the post-deployment validation step?")
    if risk and risk.notes.get("contradiction"):
        questions.append("Why does the declared risk level not match the described impact?")
    if not questions:
        questions.append("Are there any residual concerns not captured in the CR fields?")
    return questions[:4]


def _recommendations(cr: dict[str, Any], findings: list[Any]) -> list[str]:
    seen: list[str] = []
    for finding in findings:
        rec = getattr(finding, "recommendation", "") or ""
        if rec and rec not in seen:
            seen.append(rec)
    recommendations = seen[:3]
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
    """Render the final CAB-facing checklist block."""
    by_name = {result.agent: result for result in agent_results}
    clone_agent = by_name.get("clone")
    clone_candidates = clone_agent.notes.get("clone_candidates", []) if clone_agent else []
    top_clone = clone_candidates[0] if clone_candidates else None

    prediction_label = {
        Decision.PASS: "✅ CAB READY",
        Decision.CONDITIONAL: "⚠️ CONDITIONAL - REVIEW REQUIRED",
        Decision.NOT_READY: "❌ NOT READY",
    }[decision]

    lines = [
        "PRE-CAB RESULT",
        "─" * 34,
        f"CR: {cr_number}",
        "",
        f"Prediction: {prediction_label}",
        f"Confidence: {confidence:.0%}",
        "",
    ]
    lines.extend(_checklist(cr, agent_results, top_clone))

    lines.append("")
    lines.append("UAT:")
    uat_line, uat_reason = _uat_summary(findings)
    lines.append(f"   {uat_line}")
    if uat_reason:
        lines.append("   Reason:")
        for wrapped in _wrap(uat_reason, 70):
            lines.append(f"   {wrapped}")

    lines.append("")
    lines.append("Historical analysis:")
    if top_clone:
        lines.append(f"   {len(clone_candidates)} similar historical change(s) identified.")
        if top_clone.get("changed_fields"):
            lines.append(f"   Difference vs. strongest match: {', '.join(top_clone['changed_fields'][:3])}.")
        else:
            lines.append("   No material decision-changing difference was found.")
    else:
        lines.append("   No similar historical changes were identified.")

    risk = by_name.get("risk")
    lines.append("")
    lines.append("Risk:")
    risk_value = _text(cr.get("Risk")) or "unknown"
    lines.append(f"   Declared risk = {risk_value}.")
    if risk and risk.notes.get("contradiction"):
        lines.append("   Contradiction detected: declared risk does not match the described impact.")
    else:
        lines.append("   No contradiction with the described implementation impact detected.")

    lines.append("")
    lines.append("Potential CAB questions:")
    for question in _cab_questions(cr, agent_results, brain_payload):
        lines.append(f"   • {question}")

    lines.append("")
    lines.append("Recommendation:")
    for rec in _recommendations(cr, findings):
        lines.append(f"   → {rec}")

    return "\n".join(lines)


def _uat_summary(findings: list[Any]) -> tuple[str, str]:
    uat_context = next((f for f in findings if getattr(f, "code", "") == "UAT_CONTEXT"), None)
    uat_not_mandatory = next((f for f in findings if getattr(f, "code", "") == "UAT_NOT_MANDATORY"), None)
    if uat_context:
        return "⚠️ Required for this change.", uat_context.technical_detail
    if uat_not_mandatory:
        return "✅ Not mandatory for this change.", uat_not_mandatory.technical_detail
    return "No UAT determination available.", ""


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines or [""]
