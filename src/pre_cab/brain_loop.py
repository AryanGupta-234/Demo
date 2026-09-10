"""Adaptive GPT-OSS 120B reasoning loop.

Free mode uses one structured synthesis call containing self-critique. Paid/benchmark mode can use a
second independent adversarial pass for higher scrutiny.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .brain import build_reasoning_payload, build_reasoning_system_prompt, self_critique_questions
from .brain_schema import BRAIN_RESPONSE_SCHEMA
from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider, ModelResponse
from .retrieval import build_cr_query
from .sanitization import sanitize_cr_for_reasoning, sanitize_value
from .schemas import AgentContext, Finding


@dataclass(frozen=True)
class ReasoningLoopResult:
    initial: ModelResponse
    critique: ModelResponse | None
    retrieved_memory: tuple[dict[str, Any], ...]
    questions: tuple[str, ...]
    mode: str = "single"


class AgenticReasoningLoop:
    """Reason over shared CR context and optionally run an independent critique pass."""

    def __init__(
        self,
        model: ModelProvider,
        memory: UnifiedMemory | None = None,
        limit: int = 12,
        mode: str | None = None,
    ) -> None:
        self.model = model
        self.memory = memory
        self.limit = limit
        self.mode = (mode or os.getenv("PRE_CAB_REASONING_MODE") or "single").strip().lower()
        if self.mode not in {"single", "dual"}:
            raise ValueError("PRE_CAB_REASONING_MODE must be 'single' or 'dual'")

    def _memories(self, cr: dict[str, Any]) -> list[Any]:
        if not self.memory:
            return []
        return self.memory.search(
            build_cr_query(cr),
            kinds=[
                MemoryKind.FACT,
                MemoryKind.CAB_HISTORY,
                MemoryKind.POLICY,
                MemoryKind.EVIDENCE,
                MemoryKind.SIMILARITY,
                MemoryKind.EPISODE,
            ],
            limit=self.limit,
        )

    def run(self, context: AgentContext, findings: list[Finding] | None = None) -> ReasoningLoopResult:
        clean_cr = sanitize_cr_for_reasoning(context.cr)
        memories = self._memories(clean_cr)
        payload = build_reasoning_payload(clean_cr, memories=memories, prior_findings=findings or [])
        payload = sanitize_value(payload)
        payload["strictness"] = context.strictness.value
        payload["self_critique_questions"] = list(self_critique_questions())
        payload["instruction"] = (
            "Produce the final structured assessment. Challenge your own assumptions inside the "
            "self_critique field before choosing the prediction. Never invent missing evidence."
        )

        initial = self.model.generate(
            system=build_reasoning_system_prompt(),
            user=json.dumps(payload, ensure_ascii=False, default=str),
            temperature=0.05,
            response_format=BRAIN_RESPONSE_SCHEMA,
            reasoning_effort="high",
        )

        critique: ModelResponse | None = None
        if self.mode == "dual":
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
                    + " You are an independent adversarial reviewer. Return the same JSON schema, but "
                    "revise the prediction when unsupported assumptions or false-pass risk warrant it."
                ),
                user=json.dumps(critique_payload, ensure_ascii=False, default=str),
                temperature=0.0,
                response_format=BRAIN_RESPONSE_SCHEMA,
                reasoning_effort="high",
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
        return ReasoningLoopResult(
            initial=initial,
            critique=critique,
            retrieved_memory=memory_view,
            questions=self_critique_questions(),
            mode=self.mode,
        )
