"""Structured reasoning brain for Pre-CAB analysis.

The brain separates facts, inferences, uncertainties and recommendations. It builds a
provider-neutral prompt payload for GPT-OSS 120B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import MemoryRecord
from .schemas import Finding


@dataclass(frozen=True)
class Hypothesis:
    name: str
    value: Any
    rationale: str
    confidence: float


@dataclass
class ReasoningState:
    hypotheses: list[Hypothesis] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def build_reasoning_payload(
    cr: dict[str, Any],
    *,
    memories: list[MemoryRecord] | None = None,
    prior_findings: list[Finding] | None = None,
) -> dict[str, Any]:
    """Create compact, auditable context for the reasoning model."""
    memories = memories or []
    prior_findings = prior_findings or []
    return {
        "cr": cr,
        "retrieved_memory": [
            {
                "id": m.memory_id,
                "kind": m.kind.value,
                "text": m.text,
                "metadata": m.metadata,
                "score": m.score,
            }
            for m in memories
        ],
        "prior_findings": [
            {
                "code": f.code,
                "severity": f.severity.value,
                "message": f.message,
                "technical_detail": f.technical_detail,
            }
            for f in prior_findings
        ],
        "reasoning_contract": {
            "facts": "directly observed from CR, memory, or evidence",
            "inferences": "conclusions derived from facts",
            "uncertainties": "information that cannot be verified",
            "contradictions": "claims conflicting with evidence or other fields",
            "recommendations": "actions that could move the CR toward readiness",
        },
    }


def self_critique_questions() -> tuple[str, ...]:
    return (
        "What evidence would make the current conclusion wrong?",
        "Which CR claims are unverified rather than proven?",
        "Did I assume UAT is required without a contextual reason?",
        "Did I accept a rollback statement without a credible recovery mechanism?",
        "Does a similar historical CR differ in any decision-relevant way?",
        "Is there any contradiction between the CR and supporting evidence?",
        "What does the CAB reviewer still need to know or ask?",
    )


def build_reasoning_system_prompt() -> str:
    return (
        "You are the reasoning brain of a Pre-CAB validator for Normal ServiceNow Change Requests. "
        "Reason over structured CR facts, retrieved organizational memory, and evidence summaries. "
        "Separate FACTS, INFERENCES, UNCERTAINTIES, CONTRADICTIONS and RECOMMENDATIONS. "
        "UAT is contextual, not universal. A credible rollback/recovery mechanism may pass even when "
        "its procedure is brief. Similar historical CRs may suggest reusable patterns, but every new "
        "CR must be delta-validated. Never invent approvals, testing, evidence, CAB outcomes, or "
        "policy requirements. Challenge your own initial conclusion before producing the final "
        "recommendation. Include both a CAB-readable explanation and relevant technical reasoning."
    )
