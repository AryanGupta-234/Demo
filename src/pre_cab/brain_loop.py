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
from .context_compaction import compact_cr, compact_evidence, compact_findings, compact_memory
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
    """Reason over shared CR/evidence context and optionally run an independent critique pass."""

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
        clean_cr = compact_cr(sanitize_cr_for_reasoning(context.cr))
        memories = self._memories(clean_cr)
        payload = build_reasoning_payload(clean_cr, memories=memories, prior_findings=findings or [])
        payload = sanitize_value(payload)
        payload["cr"] = compact_cr(dict(payload.get("cr") or {}))
        payload["retrieved_memory"] = compact_memory(list(payload.get("retrieved_memory") or []), limit=8)
        payload["prior_findings"] = compact_findings(list(payload.get("prior_findings") or []), limit=24)
        payload["strictness"] = context.strictness.value
        payload["evidence"] = compact_evidence(sanitize_value(list(context.evidence or ())), limit=10)
        payload["self_critique_questions"] = list(self_critique_questions())
        payload["instruction"] = (
            "Produce the final structured assessment using CR fields, retrieved history/policy, and "
            "attachment evidence. Challenge your own assumptions inside self_critique before choosing "
            "the prediction. Never invent missing evidence. Evidence contradictions override unsupported "
            "CR claims."
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
