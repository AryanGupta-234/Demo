"""Structured reasoning brain for Pre-CAB analysis.

The brain separates facts, inferences, uncertainties and recommendations and
builds provider-neutral context for the active local reasoning model.
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
            "facts": "directly observed from the current CR, Work Notes, memory, or evidence",
            "inferences": "conclusions derived from those facts",
            "uncertainties": "information that cannot be verified",
            "contradictions": "claims conflicting with evidence, Work Notes, or other fields",
            "recommendations": "actions that could move the CR toward readiness",
        },
    }


def self_critique_questions() -> tuple[str, ...]:
    return (
        "What evidence would make the current conclusion wrong?",
        "Which CR or Work Note claims are unverified rather than proven?",
        "Did I assume UAT is required without a contextual reason?",
        "Did I treat SIT, UAT and Pre-PROD as separate requirements when the CR only establishes non-PROD validation?",
        "Did I incorrectly convert an unset Environment field into a missing-target defect?",
        "Did I treat a missing Configuration Item as a functional or implementation failure rather than a traceability observation?",
        "Did I accept a rollback statement without a credible recovery mechanism?",
        "Did I distinguish current execution evidence from historical examples or planned testing?",
        "Did I distinguish applicable evidence from merely useful evidence?",
        "Did a single keyword drive the change classification or impact assessment without corroborating context?",
        "Does a similar historical CR differ in any decision-relevant way?",
        "Is there any contradiction between the CR, Work Notes, and supporting evidence?",
        "What does the CAB reviewer still need to know or ask?",
    )


def build_reasoning_system_prompt() -> str:
    return (
        "You are the reasoning brain of a Pre-CAB validator for Normal ServiceNow Change Requests. "
        "Validate every model-facing descriptive and signoff field, but do not confuse validation with "
        "universal mandatory population. First classify the change intent and archetype, then evaluate evidence "
        "against what that archetype actually needs. Reason over the current CR, mapped requirements, same-CR Work Notes/Comments, "
        "retrieved organizational memory, and evidence summaries. Distinguish test plan, test execution, and formal "
        "Test Results Evidence. A historical CR number mentioned in a test plan is reference material, not proof of current execution. "
        "Interpret signoffs by disposition (Yes/No/Not Applicable/etc.). UAT is contextual. "
        "SIT, UAT, Pre-PROD, staging and lower/test environment references are one NON-PROD validation class; "
        "do not create separate gates for each label. For this PROD workflow, an unset Environment is resolved "
        "from workflow context and is not a missing-target defect. A missing Configuration Item is primarily a "
        "CMDB/operational traceability observation unless explicit policy makes it a gate. Similar historical CRs "
        "are evidence for context and delta analysis, never permission to copy outcomes. Never invent approvals, "
        "testing, evidence, CAB outcomes, or policy. Do not let one generic keyword such as 'critical', 'user', "
        "'transaction', or 'customer' determine impact without context. Treat contradictory evidence as more important "
        "than missing metadata, and distinguish hard blockers, actionable warnings, and informational observations. "
        "Separate facts, inferences, uncertainties and contradictions and challenge false-PASS risk. Produce both "
        "CAB-readable and technical reasoning."
    )
