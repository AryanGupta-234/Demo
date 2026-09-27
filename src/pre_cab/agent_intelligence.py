"""Cross-agent intelligence layer for the Pre-CAB reasoning engine.

The specialist agents remain independently deterministic. This module turns their outputs
into one compact, auditable "agent council" view that the neural reasoning layer can use.
It does not make or override the final policy decision.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .agents import AgentResult
from .schemas import FindingSeverity


_SEVERITY_RANK = {
    FindingSeverity.BLOCKING.value: 3,
    FindingSeverity.WARNING.value: 2,
    FindingSeverity.INFO.value: 1,
}


def _finding_dict(finding: Any) -> dict[str, Any]:
    return {
        "code": getattr(finding, "code", ""),
        "title": getattr(finding, "title", ""),
        "severity": getattr(getattr(finding, "severity", None), "value", str(getattr(finding, "severity", ""))),
        "message": getattr(finding, "message", ""),
        "technical_detail": getattr(finding, "technical_detail", ""),
        "evidence_refs": list(getattr(finding, "evidence_refs", ()) or ()),
        "recommendation": getattr(finding, "recommendation", ""),
    }


def build_agent_intelligence(results: Iterable[AgentResult]) -> dict[str, Any]:
    """Fuse specialist outputs into a model-ready, disagreement-aware council view."""
    results = list(results)
    per_agent: dict[str, dict[str, Any]] = {}
    code_to_agents: dict[str, list[str]] = defaultdict(list)
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []

    for result in results:
        findings = [_finding_dict(item) for item in result.findings]
        for finding in findings:
            code = str(finding["code"])
            code_to_agents[code].append(result.agent)
            bucket = str(finding["severity"]).upper()
            if bucket == FindingSeverity.BLOCKING.value:
                blocking.append({**finding, "agent": result.agent})
            elif bucket == FindingSeverity.WARNING.value:
                warnings.append({**finding, "agent": result.agent})
            else:
                info.append({**finding, "agent": result.agent})

        notes = result.notes if isinstance(result.notes, dict) else {}
        chain = notes.get("chain") if isinstance(notes.get("chain"), list) else []
        per_agent[result.agent] = {
            "finding_count": len(findings),
            "max_severity": max(
                (str(item["severity"]).upper() for item in findings),
                key=lambda value: _SEVERITY_RANK.get(value, 0),
                default="NONE",
            ),
            "findings": findings[:8],
            "requirements": list(result.requirements or [])[:10],
            "chain": [str(item) for item in chain[:8]],
            "key_metrics": {
                key: value
                for key, value in notes.items()
                if key not in {"chain", "clone_candidates", "matches", "change_profile", "evidence_matrix"}
                and isinstance(value, (str, int, float, bool))
            },
        }

    repeated = [
        {"code": code, "agents": agents, "agreement_count": len(agents)}
        for code, agents in code_to_agents.items()
        if len(agents) >= 2
    ]

    # Explicit disagreement heuristics keep the neural layer from treating one agent's
    # perspective as universally authoritative.
    disagreements: list[dict[str, Any]] = []
    testing = per_agent.get("testing", {})
    technical = per_agent.get("technical", {})
    context = per_agent.get("context", {})
    risk = per_agent.get("risk", {})

    testing_chain = " ".join(testing.get("chain", [])).lower()
    technical_chain = " ".join(technical.get("chain", [])).lower()
    context_chain = " ".join(context.get("chain", [])).lower()
    risk_chain = " ".join(risk.get("chain", [])).lower()

    if "test plan is defined" in testing_chain and "testing plan missing" in testing_chain:
        disagreements.append({
            "topic": "testing_plan_presence",
            "detail": "Testing agent produced internally inconsistent plan-presence signals.",
        })
    if "formal test evidence expected = true" in context_chain and "formal test evidence is not required" in testing_chain:
        disagreements.append({
            "topic": "formal_test_evidence_applicability",
            "detail": "Context and testing agents disagree on whether formal test evidence is applicable.",
        })
    if "rollback mechanism aligns with implementation" in technical_chain and "rollback mechanism could not be correlated" in technical_chain:
        disagreements.append({
            "topic": "rollback_alignment",
            "detail": "Technical agent contains both positive and negative rollback-correlation signals; inspect the implementation/backout pair.",
        })
    if "contradiction" in risk_chain and "no direct contradiction" in risk_chain:
        disagreements.append({
            "topic": "risk_contradiction",
            "detail": "Risk agent produced mixed contradiction signals; rely on the underlying evidence rather than the summary wording.",
        })

    # Keep the council compact enough for a 7B model while retaining the strongest evidence.
    return {
        "agent_count": len(results),
        "agents": per_agent,
        "consensus": {
            "blocking_count": len(blocking),
            "warning_count": len(warnings),
            "info_count": len(info),
            "repeated_finding_codes": repeated[:20],
        },
        "blocking_findings": blocking[:12],
        "warning_findings": warnings[:16],
        "disagreements": disagreements[:12],
        "reasoning_policy": {
            "specialists_are_observers": True,
            "deterministic_gates_remain_authoritative": True,
            "repeated_findings_increase_confidence_but_do_not_create_new_policy": True,
            "disagreements_require_reinspection_of_source_fields": True,
            "model_must_not_invent_missing_evidence": True,
        },
    }
