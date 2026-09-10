"""Runtime for the shared reasoning brain.

This module prepares context for GPT-OSS 120B and exposes a deterministic self-critique checklist.
It does not make an approval decision itself.
"""
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
class BrainResult:
    response: ModelResponse
    payload: dict[str, Any]
    critique_questions: tuple[str, ...]


class ReasoningBrain:
    """Provider-neutral reasoning coordinator; V1 model is GPT-OSS 120B through the model adapter."""

    def __init__(self, model: ModelProvider, memory: UnifiedMemory | None = None) -> None:
        self.model = model
        self.memory = memory

    def _retrieve(self, cr: dict[str, Any]) -> list[Any]:
        if not self.memory:
            return []
        query = build_cr_query(cr)
        return self.memory.search(
            query,
            kinds=[MemoryKind.CAB_HISTORY, MemoryKind.POLICY, MemoryKind.EVIDENCE, MemoryKind.SIMILARITY, MemoryKind.EPISODE],
            limit=12,
        )

    def reason(self, context: AgentContext, findings: list[Finding] | None = None) -> BrainResult:
        memories = self._retrieve(context.cr)
        payload = build_reasoning_payload(context.cr, memories=memories, prior_findings=findings or [])
        payload["strictness"] = context.strictness.value
        payload["self_critique"] = list(self_critique_questions())
        response = self.model.generate(
            system=build_reasoning_system_prompt(),
            user=json.dumps(payload, ensure_ascii=False, default=str),
            temperature=0.1,
        )
        return BrainResult(response=response, payload=payload, critique_questions=self_critique_questions())
