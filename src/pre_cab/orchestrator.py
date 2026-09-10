"""Stage-1 orchestration. Stage-2 attachment verification is intentionally a separate gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import DEFAULT_AGENT_TYPES, AgentResult
from .decision import validate_fields
from .memory import UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Decision, Finding, FindingSeverity, Strictness, ValidationResult


@dataclass
class PipelineResult:
    stage1: ValidationResult
    agent_results: list[AgentResult]
    stage2_required: bool = True


def run_stage1(
    cr: dict[str, Any],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: ModelProvider | None = None,
    memory: UnifiedMemory | None = None,
) -> PipelineResult:
    """Run Normal-CR field/context analysis and prepare a Stage-2 evidence gate."""
    stage1 = validate_fields(cr, strictness)
    context = AgentContext(cr=cr, strictness=strictness)
    results: list[AgentResult] = []

    # Keep the shared context simple in V1. Future agents can consume earlier structured findings.
    for agent_type in DEFAULT_AGENT_TYPES:
        agent = agent_type(model=model, memory=memory)
        results.append(agent.run(context))

    # Aggregate agent findings without allowing LLM text to silently bypass hard gates.
    merged = list(stage1.findings)
    for result in results:
        merged.extend(result.findings)

    blocking = [f for f in merged if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in merged if f.severity == FindingSeverity.WARNING]
    if blocking:
        decision = Decision.NOT_READY
    elif warnings:
        decision = Decision.CONDITIONAL
    else:
        decision = Decision.PASS

    stage1.findings = merged
    stage1.decision = decision
    stage1.metadata.update(
        {
            "stage": 1,
            "stage_2_required": decision != Decision.NOT_READY,
            "agents": [r.agent for r in results],
        }
    )
    return PipelineResult(stage1=stage1, agent_results=results, stage2_required=decision != Decision.NOT_READY)
