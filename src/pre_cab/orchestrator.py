"""Pre-CAB orchestration: deterministic gates + specialist agents + reasoning brain."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agents import DEFAULT_AGENT_TYPES, AgentResult
from .brain_loop import AgenticReasoningLoop, ReasoningLoopResult
from .decision import validate_fields
from .memory import UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Decision, Finding, FindingSeverity, Strictness, ValidationResult


@dataclass
class PipelineResult:
    stage1: ValidationResult
    agent_results: list[AgentResult]
    stage2_required: bool = True
    reasoning: ReasoningLoopResult | None = None


_DECISION_RANK = {
    Decision.PASS: 0,
    Decision.CONDITIONAL: 1,
    Decision.NOT_READY: 2,
}


def _parse_brain_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _brain_decision(reasoning: ReasoningLoopResult | None) -> tuple[Decision | None, dict[str, Any] | None]:
    if reasoning is None:
        return None, None
    # In dual mode, the independent critique is the final model recommendation.
    response = reasoning.critique or reasoning.initial
    payload = _parse_brain_payload(response.text)
    if not payload:
        return None, None
    try:
        return Decision(str(payload.get("prediction"))), payload
    except ValueError:
        return None, payload


def _more_conservative(a: Decision, b: Decision | None) -> Decision:
    if b is None:
        return a
    return b if _DECISION_RANK[b] > _DECISION_RANK[a] else a


def run_stage1(
    cr: dict[str, Any],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: ModelProvider | None = None,
    memory: UnifiedMemory | None = None,
) -> PipelineResult:
    """Run deterministic validation, specialist agents, retrieval and optional GPT-OSS reasoning.

    The reasoning model may make the result more conservative, but it can never override a
    deterministic blocker and upgrade the decision. This preserves agentic prediction while keeping
    governance-critical gates auditable.
    """
    stage1 = validate_fields(cr, strictness)
    context = AgentContext(cr=cr, strictness=strictness)
    results: list[AgentResult] = []

    for agent_type in DEFAULT_AGENT_TYPES:
        agent = agent_type(model=model, memory=memory)
        results.append(agent.run(context))

    merged = list(stage1.findings)
    for result in results:
        merged.extend(result.findings)

    blocking = [f for f in merged if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in merged if f.severity == FindingSeverity.WARNING]
    deterministic_decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)

    reasoning: ReasoningLoopResult | None = None
    if model is not None:
        brain = AgenticReasoningLoop(model=model, memory=memory)
        reasoning = brain.run(context, findings=merged)

    model_decision, brain_payload = _brain_decision(reasoning)
    decision = _more_conservative(deterministic_decision, model_decision)

    if model_decision is not None and _DECISION_RANK[model_decision] > _DECISION_RANK[deterministic_decision]:
        reason_text = "GPT-OSS identified additional risk not represented by the deterministic gates."
        recommendation = "Review the model's CAB/technical reasoning and supporting evidence before approval."
        if brain_payload:
            cab_reasoning = str(brain_payload.get("cab_reasoning") or "").strip()
            recommendations = brain_payload.get("recommendations") or []
            if cab_reasoning:
                reason_text = cab_reasoning
            if isinstance(recommendations, list) and recommendations:
                recommendation = "; ".join(str(item) for item in recommendations[:3])
        merged.append(
            Finding(
                code="MODEL_CONSERVATIVE_DOWNGRADE",
                title="Reasoning model identified additional CAB risk",
                severity=(FindingSeverity.BLOCKING if model_decision == Decision.NOT_READY else FindingSeverity.WARNING),
                message=reason_text,
                technical_detail="The model recommendation can downgrade but never upgrade deterministic readiness.",
                recommendation=recommendation,
            )
        )

    stage1.findings = merged
    stage1.decision = decision
    if brain_payload:
        confidence = brain_payload.get("confidence")
        if isinstance(confidence, (int, float)):
            # Keep confidence conservative by never exceeding either source's confidence.
            stage1.confidence = min(stage1.confidence, max(0.0, min(1.0, float(confidence))))
        stage1.technical_summary = str(brain_payload.get("technical_reasoning") or stage1.technical_summary)
        stage1.cab_summary = str(brain_payload.get("cab_reasoning") or stage1.cab_summary)

    stage1.metadata.update(
        {
            "stage": 1,
            "stage_2_required": decision != Decision.NOT_READY,
            "agents": [result.agent for result in results],
            "reasoning_mode": reasoning.mode if reasoning is not None else "none",
            "reasoning_passes": (2 if reasoning and reasoning.critique is not None else (1 if reasoning else 0)),
            "model": getattr(model, "model_name", None),
            "model_prediction": model_decision.value if model_decision else None,
            "deterministic_prediction": deterministic_decision.value,
            "memory_records_retrieved": len(reasoning.retrieved_memory) if reasoning else 0,
            "brain": brain_payload or {},
        }
    )
    return PipelineResult(
        stage1=stage1,
        agent_results=results,
        stage2_required=decision != Decision.NOT_READY,
        reasoning=reasoning,
    )
