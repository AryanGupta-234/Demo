"""Adaptive GPT-OSS 120B reasoning loop using robust JSON-object output."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .brain import build_reasoning_payload, build_reasoning_system_prompt, self_critique_questions
from .context_compaction import compact_cr, compact_evidence, compact_findings, compact_memory
from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider, ModelResponse
from .retrieval import build_cr_query
from .sanitization import sanitize_cr_for_reasoning, sanitize_value
from .schemas import AgentContext, Finding

_JSON_OBJECT_FORMAT = {"type": "json_object"}


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
        effort = (os.getenv("PRE_CAB_REASONING_EFFORT") or "medium").strip().lower()
        if effort not in {"low", "medium", "high"}:
            effort = "medium"
        self.reasoning_effort = effort

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

    def _build_payload(self, context: AgentContext, findings: list[Finding] | None) -> dict[str, Any]:
        # Field Agent selects the model-facing CR. Deterministic agents retain the
        # full normalized record through context.cr; GPT sees only decision-relevant
        # fields to control token use and avoid noise from dozens of unused columns.
        selected_cr = context.llm_cr if isinstance(context.llm_cr, dict) and context.llm_cr else context.cr
        clean_cr = compact_cr(sanitize_cr_for_reasoning(selected_cr))
        memories = self._memories(clean_cr)
        payload = build_reasoning_payload(clean_cr, memories=memories, prior_findings=findings or [])
        payload = sanitize_value(payload)
        payload["cr"] = compact_cr(dict(payload.get("cr") or {}))
        payload["retrieved_memory"] = compact_memory(list(payload.get("retrieved_memory") or []), limit=8)
        payload["prior_findings"] = compact_findings(list(payload.get("prior_findings") or []), limit=18)
        payload["strictness"] = context.strictness.value
        payload["field_selection"] = {
            "selected_field_count": len(selected_cr),
            "selected_fields": list(selected_cr.keys()),
            "selection_source": "field-agent-policy-and-historical-intelligence",
        }
        payload["evidence"] = compact_evidence(sanitize_value(list(context.evidence or ())), limit=8)
        payload["self_critique_questions"] = list(self_critique_questions())
        payload["output_contract"] = {
            "prediction": "PASS, CONDITIONAL, or NOT_READY",
            "confidence": "number from 0 to 1",
            "facts": "array of concise factual observations only",
            "inferences": "array of reasoned conclusions grounded in facts",
            "uncertainties": "array of unresolved items",
            "contradictions": "array of detected inconsistencies",
            "technical_reasoning": "substantive technical assessment in a few sentences",
            "cab_reasoning": "plain-language CAB decision rationale in a few sentences",
            "cab_questions": "array of up to 4 useful CAB questions",
            "recommendations": "array of up to 4 concrete next actions",
            "self_critique": "array of concise checks used to challenge the conclusion",
        }
        payload["instruction"] = (
            "Return ONLY one valid JSON object. Do not use markdown. Include every key named in "
            "output_contract. Keep arrays concise and reasoning substantive but bounded. Analyze the actual "
            "CR and specialist findings; do not merely restate field presence. Explain why the change is or "
            "is not ready. Use historical memory and clone information when available. The CR shown here is "
            "intentionally field-selected; do not infer that omitted fields are missing. UAT is contextual, "
            "not universal. A rollback is acceptable only when it represents a credible recovery mechanism. "
            "Never invent approvals, testing, evidence, history, or policy. Distinguish facts, inferences, and "
            "uncertainties. Challenge the conclusion for false-PASS risk before finalizing."
        )
        return payload

    def run(self, context: AgentContext, findings: list[Finding] | None = None) -> ReasoningLoopResult:
        payload = self._build_payload(context, findings)
        initial = self.model.generate(
            system=build_reasoning_system_prompt(),
            user=json.dumps(payload, ensure_ascii=False, default=str),
            temperature=0.05,
            response_format=_JSON_OBJECT_FORMAT,
            reasoning_effort=self.reasoning_effort,
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
                    "return_only_valid_json": True,
                },
            }
            critique = self.model.generate(
                system=(
                    build_reasoning_system_prompt()
                    + " You are an independent adversarial reviewer. Return one valid JSON object "
                    "using the same output contract and revise the prediction when warranted. Keep the "
                    "response concise enough to fit the completion budget."
                ),
                user=json.dumps(critique_payload, ensure_ascii=False, default=str),
                temperature=0.0,
                response_format=_JSON_OBJECT_FORMAT,
                reasoning_effort=self.reasoning_effort,
            )

        selected_cr = context.llm_cr if isinstance(context.llm_cr, dict) and context.llm_cr else context.cr
        clean_cr = compact_cr(sanitize_cr_for_reasoning(selected_cr))
        memories = self._memories(clean_cr)
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
