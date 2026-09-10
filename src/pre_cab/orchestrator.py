"""Pre-CAB orchestration: deterministic gates + specialist agents + reasoning brain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import DEFAULT_AGENT_TYPES, AgentResult
from .brain_loop import AgenticReasoningLoop, ReasoningLoopResult
from .decision import validate_fields
from .memory import UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Decision, FindingSeverity, Strictness, ValidationResult


@dataclass
class PipelineResult:
    stage1: ValidationResult
    agent_results: list[AgentResult]
    stage2_required: bool = True
    reasoning: ReasoningLoopResult | None = None


def run_stage1(
    cr: dict[str, Any],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: ModelProvider | None = None,
    memory: UnifiedMemory | None = None,
) -> PipelineResult:
    """Run field validation, specialist agents, shared memory and optional two-pass reasoning."""
    stage1 = validate_fields(cr, strictness)
    context = AgentContext(cr=cr, strictness=strictness)
    results: list[AgentResult] = []

    # Specialist agents share the same model/memory interfaces and produce typed findings.
    for agent_type in DEFAULT_AGENT_TYPES:
        agent = agent_type(model=model, memory=memory)
        results.append(agent.run(context))

    merged = list(stage1.findings)
    for result in results:
        merged.extend(result.findings)

    reasoning: ReasoningLoopResult | None = None
    if model is not None:
        brain = AgenticReasoningLoop(model=model, memory=memory)
        reasoning = brain.run(context, findings=merged)

    blocking = [f for f in merged if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in merged if f.severity == FindingSeverity.WARNING]
    decision = Decision.NOT_READY if blocking else (Decision.CONDITIONAL if warnings else Decision.PASS)

    stage1.findings = merged
    stage1.decision = decision
    stage1.metadata.update(
        {
            "stage": 1,
            "stage_2_required": decision != Decision.NOT_READY,
            "agents": [r.agent for r in results],
            "reasoning_passes": 2 if reasoning is not None else 0,
            "model": getattr(model, "model_name", None),
            "memory_records_retrieved": len(reasoning.retrieved_memory) if reasoning else 0,
        }
    )
    return PipelineResult(
        stage1=stage1,
        agent_results=results,
        stage2_required=decision != Decision.NOT_READY,
        reasoning=reasoning,
    )
