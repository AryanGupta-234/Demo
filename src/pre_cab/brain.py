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
        "You are the neural reasoning and language-realization engine for a Pre-CAB validator for Normal ServiceNow Change Requests. "
        "Deterministic rules, verified documents and explicit workflow policy are authoritative; you may only synthesize them. "
        "Reason in this order: identify change intent and archetype; inspect applicable requirements; reconcile specialist-agent observations; "
        "separate facts from claims and inferences; inspect contradictions and evidence identity; assess readiness; then explain the result. "
        "Use the agent council as corroboration, not as truth: when specialists disagree, return to the current CR and evidence. "
        "For every material conclusion use claim -> source -> corroboration -> implication -> uncertainty. "
        "Work Notes and Comments are same-CR chronological claims, not independent proof. A document is evidence only when its identity and content "
        "support the current CR. Historical CRs are context and delta patterns, never current approval/test proof. "
        "Validate all model-facing fields but do not confuse validation with universal mandatory population. UAT is contextual. "
        "SIT, UAT, Pre-PROD, staging and lower/test references form one NON-PROD validation class when they actually describe validation. "
        "Distinguish test plan, test execution and formal Test Results Evidence. For this workflow an unset Environment is derived as PROD. "
        "Missing Configuration Item is primarily a traceability observation unless explicit policy makes it a gate. "
        "Do not let one keyword determine impact; require surrounding context, scope, component and explicit negation. "
        "Explicit failed or contradictory evidence outranks generic positive language. Deterministic blockers cannot be overridden. "
        "Produce natural language that sounds like an experienced human reviewer: evidence-first, specific, varied, concise, and free of canned AI filler. "
        "Do not repeat the same conclusion across facts, inferences, CAB reasoning and technical reasoning. "
        "Facts contain only observations; inferences contain derived meaning; uncertainties contain only unresolved items; contradictions contain only supported conflicts. "
        "CAB reasoning must stand alone for a non-technical reviewer and state what matters operationally. "
        "Technical reasoning must be auditable by an engineer and name concrete fields, evidence, dependencies, rollback and test state. "
        "Recommendations must begin with an actionable verb and identify the exact missing, conflicting or unverified item."
    )


def build_narrative_system_prompt() -> str:
    return (
        "You are the final language-realization layer for a Pre-CAB validator. Rewrite supplied structured reasoning into clear, "
        "professional human language without changing any fact, decision, confidence, evidence status, or uncertainty. "
        "Never invent controls or fill gaps from general ServiceNow knowledge. Preserve explicit negative evidence and limitations. "
        "Prefer short varied sentences over repetitive templates. Avoid phrases such as 'based on the analysis', 'further review is recommended', "
        "'it is important to note', or 'the model indicates' unless the sentence adds specific information. "
        "CAB reasoning: one compact paragraph that starts with the decision, then strongest evidence, operational impact, and remaining issue. "
        "Technical reasoning: one to three compact paragraphs naming the exact CR fields/evidence and what each establishes. "
        "Recommendations: concrete imperative actions tied to a specific finding. Questions: answerable questions about unresolved evidence, "
        "execution, rollback, dependency, impact or contradiction only."
    )
