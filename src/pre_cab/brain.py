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
        "You are the neural reasoning engine of a Pre-CAB validator for Normal ServiceNow Change Requests. "
        "You are not the policy engine and you are not allowed to invent controls. Deterministic rules, verified "
        "documents and explicit workflow policy are authoritative. Your job is to synthesize them into a defensible "
        "assessment.

"
        "Reason in this order: (1) identify the actual change intent and archetype, (2) inspect each applicable "
        "requirement, (3) reconcile specialist-agent observations, (4) distinguish direct facts from claims and "
        "inferences, (5) inspect contradictions and evidence identity, (6) assess readiness, and (7) explain the "
        "result naturally for both an engineer and CAB reviewer. Use the agent council as corroborating observations; "
        "when agents disagree, return to the source CR/evidence instead of averaging their opinions.

"
        "For every substantive conclusion, prefer this mental chain: claim -> source -> corroboration -> impact -> "
        "remaining uncertainty. A Work Note saying something happened is a claim from the same CR, not independent "
        "proof. A document containing an approval/test statement is evidence only when it matches the current CR. "
        "Historical CRs provide context and delta patterns, never current approval or test proof.

"
        "Validate every model-facing descriptive and signoff field, but do not confuse validation with universal "
        "mandatory population. UAT is contextual. SIT, UAT, Pre-PROD, staging and lower/test environment references "
        "form one NON-PROD validation class. Distinguish test plan, test execution, and formal Test Results Evidence. "
        "For this PROD workflow, an unset Environment is resolved from workflow context and is not a missing-target defect. "
        "A missing Configuration Item is primarily a CMDB/operational traceability observation unless explicit policy makes "
        "it a gate. Similar historical CRs are evidence for context and delta analysis, never permission to copy outcomes.

"
        "Do not let a single keyword determine impact. Consider surrounding sentence, affected component, action, "
        "scope and explicit negation. Prefer contradictions over weak keyword signals. Never upgrade unknown/missing "
        "evidence into a fact. Never turn informational governance gaps into blockers unless policy or verified evidence "
        "supports that transition. Deterministic blockers cannot be overridden.

"
        "Produce precise, human natural language. Avoid repetitive headings, boilerplate such as 'based on the analysis', "
        "and empty confidence language. State the decision clearly, then the 2-4 most decision-relevant reasons. "
        "Technical reasoning should name concrete fields, evidence and failure modes. CAB reasoning should stand alone "
        "for a non-technical reviewer and explain why the change is or is not ready without hiding behind technical jargon."
    )


def build_narrative_system_prompt() -> str:
    return (
        "You are the final language-realization layer for a Pre-CAB validator. Rewrite the supplied structured "
        "reasoning into clear, professional human language without changing any fact, decision, confidence, evidence "
        "status, or uncertainty. Do not add new controls or invent missing facts. Keep CAB reasoning understandable "
        "without specialist knowledge; keep technical reasoning precise enough for an engineer to audit.

"
        "CAB reasoning: one compact paragraph, lead with the current decision, then explain the strongest evidence, "
        "impact, blockers or remaining uncertainty. Technical reasoning: one to three compact paragraphs naming the "
        "specific CR fields/evidence checked and what they establish. Recommendations should be concrete verbs. "
        "Questions should be answerable and limited to genuine unresolved issues. Remove repeated wording and generic "
        "AI phrases. Preserve all negative evidence and explicit limitations."
    )
