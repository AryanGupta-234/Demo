"""Two-pass reasoning loop: initial analysis followed by adversarial self-critique."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .brain import build_reasoning_payload, build_reasoning_system_prompt, self_critique_questions
from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider, ModelResponse
from .retrieval import build_cr_query
from .schemas import AgentContext, Finding


@dataclass(frozen=True)
class ReasoningLoopResult:
    initial: ModelResponse
    critique: ModelResponse
    retrieved_memory: tuple[dict[str, Any], ...]
    questions: tuple[str, ...]


class AgenticReasoningLoop:
    """Run an initial reasoning pass and an independent critique pass with shared context."""

    def __init__(self, model: ModelProvider, memory: UnifiedMemory | None = None, limit: int = 12) -> None:
        self.model = model
        self.memory = memory
        self.limit = limit

    def _memories(self, cr: dict[str, Any]) -> list[Any]:
        if not self.memory:
            return []
        return self.memory.search(
            build_cr_query(cr),
            kinds=[MemoryKind.FACT, MemoryKind.CAB_HISTORY, MemoryKind.POLICY, MemoryKind.EVIDENCE, MemoryKind.SIMILARITY, MemoryKind.EPISODE],
            limit=self.limit,
        )

    def run(self, context: AgentContext, findings: list[Finding] | None = None) -> ReasoningLoopResult:
        memories = self._memories(context.cr)
        payload = build_reasoning_payload(context.cr, memories=memories, prior_findings=findings or [])
        payload["strictness"] = context.strictness.value
        payload["self_critique_questions"] = list(self_critique_questions())

        initial = self.model.generate(
            system=build_reasoning_system_prompt(),
            user=json.dumps(payload, ensure_ascii=False, default=str),
            temperature=0.1,
        )

        critique_payload = {
            "original_context": payload,
            "initial_reasoning": initial.text,
            "critique_contract": {
                "challenge_every_inference": True,
                "never_upgrade_unknown_to_fact": True,
                "never_invent_evidence": True,
                "look_for_contradictions": True,
                "look_for_false_pass_risk": True,
            },
        }
        critique = self.model.generate(
            system=(
                build_reasoning_system_prompt()
                + " You are now the adversarial reviewer of the first pass. "
                "Identify unsupported assumptions, missing evidence, contradictions, and reasons "
                "the proposed result could be a false PASS. State what should change."
            ),
            user=json.dumps(critique_payload, ensure_ascii=False, default=str),
            temperature=0.05,
        )
        memory_view = tuple(
            {
                "id": m.memory_id,
                "kind": m.kind.value,
                "text": m.text,
                "metadata": m.metadata,
                "score": m.score,
            }
            for m in memories
        )
        return ReasoningLoopResult(initial=initial, critique=critique, retrieved_memory=memory_view, questions=self_critique_questions())
