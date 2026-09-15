"""Adaptive local reasoning loop for the Pre-CAB validator."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brain import build_reasoning_payload, build_reasoning_system_prompt, self_critique_questions
from .context_compaction import compact_cr, compact_evidence, compact_findings, compact_memory
from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider, ModelResponse
from .retrieval import build_cr_query
from .sanitization import sanitize_cr_for_reasoning, sanitize_value
from .schemas import AgentContext, Finding

_JSON_OBJECT_FORMAT = {"type": "json_object"}

_DESCRIPTIVE_FIELDS = (
    "Short description", "Description", "Justification", "Implementation plan",
    "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
    "Configuration item", "Risk", "Priority",
)
_SIGNOFF_FIELDS = (
    "UAT signoff", "Customer Approval", "TCS QA signoff",
    "Test Results Evidence", "Lower Environment Reference CR/SR",
)
_CONTEXT_FIELDS = (
    "Number", "Type", "Category", "Sub Category", "Environment",
    "Planned start", "Planned end", "Conflict status", "Change Class",
)
_LEGACY_COMBINED_NOTES = "Comments and Work notes"
_REQUIREMENT_CONTEXT_FIELDS = ("Priority", "Risk and impact analysis", "Lower Environment Reference CR/SR", "TCS QA signoff")


@dataclass(frozen=True)
class ReasoningLoopResult:
    initial: ModelResponse
    critique: ModelResponse | None
    retrieved_memory: tuple[dict[str, Any], ...]
    questions: tuple[str, ...]
    mode: str = "single"


class AgenticReasoningLoop:
    """Reason over the mapped CR, its own Work Notes/Comments, requirements and history."""

    def __init__(self, model: ModelProvider, memory: UnifiedMemory | None = None, limit: int = 12, mode: str | None = None) -> None:
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
            kinds=[MemoryKind.FACT, MemoryKind.CAB_HISTORY, MemoryKind.POLICY, MemoryKind.EVIDENCE, MemoryKind.SIMILARITY, MemoryKind.EPISODE],
            limit=self.limit,
        )

    @staticmethod
    def _mapped_cr(cr: dict[str, Any]) -> dict[str, Any]:
        mapped: dict[str, Any] = {}
        for field in _DESCRIPTIVE_FIELDS + _SIGNOFF_FIELDS:
            mapped[field] = cr.get(field) if field in cr else None
        for field in _CONTEXT_FIELDS:
            if cr.get(field) not in (None, "", [], {}):
                mapped[field] = cr[field]
        return mapped

    @staticmethod
    def _journal_context(cr: dict[str, Any]) -> dict[str, Any]:
        value: dict[str, Any] = {
            "work_notes": str(cr.get("Work notes") or "").strip() or None,
            "comments": str(cr.get("Comments") or "").strip() or None,
        }
        if cr.get(_LEGACY_COMBINED_NOTES) not in (None, "", [], {}):
            value["legacy_comments_and_work_notes"] = str(cr.get(_LEGACY_COMBINED_NOTES) or "").strip()
        return value

    @staticmethod
    def _historical_context() -> dict[str, Any] | None:
        raw = os.getenv("PRE_CAB_TRAINING_CONTEXT", "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {"status": "invalid", "path": str(path)}
        except Exception as exc:
            return {"status": "unreadable", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _compact_historical_context(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if not value:
            return None
        records = []
        for record in list(value.get("records") or []):
            if isinstance(record, dict):
                records.append({
                    "cr": record.get("cr", {}),
                    "requirements": record.get("requirements", {}),
                    "work_notes": str(record.get("work_notes") or "")[:800],
                    "comments": str(record.get("comments") or "")[:800],
                })
        return {
            "version": value.get("version"),
            "records_mapped": value.get("records_mapped", len(records)),
            "context_buckets": list(value.get("context_buckets") or []),
            "historical_note_phrases": list(value.get("historical_note_phrases") or [])[:40],
            "records": records,
        }

    def _build_payload(self, context: AgentContext, findings: list[Finding] | None) -> dict[str, Any]:
        base = context.llm_cr if isinstance(context.llm_cr, dict) and context.llm_cr else context.cr
        selected_cr = dict(base)
        # Always expose the explicit CR schema requested by the training mapper.
        # This prevents the field-agent's historical sparsity filter from dropping
        # a field whose presence/absence is itself useful to Qwen.
        selected_cr.update(self._mapped_cr(context.cr))
        for field in _REQUIREMENT_CONTEXT_FIELDS:
            if field in context.cr and context.cr.get(field) not in (None, "", [], {}):
                selected_cr[field] = context.cr[field]
        clean_cr = compact_cr(sanitize_cr_for_reasoning(selected_cr))
        memories = self._memories(clean_cr)
        payload = sanitize_value(build_reasoning_payload(clean_cr, memories=memories, prior_findings=findings or []))
        payload["cr"] = compact_cr(dict(payload.get("cr") or {}))
        journal = self._journal_context(context.cr)
        payload["mapped_descriptive_fields"] = list(_DESCRIPTIVE_FIELDS)
        payload["mapped_signoff_fields"] = list(_SIGNOFF_FIELDS)
        payload["mapped_context_fields"] = list(_CONTEXT_FIELDS)
        payload["descriptive_field_presence"] = {field: selected_cr.get(field) not in (None, "", [], {}) for field in _DESCRIPTIVE_FIELDS}
        payload["signoff_dispositions"] = {field: str(selected_cr.get(field)).strip() if selected_cr.get(field) not in (None, "", [], {}) else None for field in _SIGNOFF_FIELDS}
        payload["work_notes"] = journal.get("work_notes")
        payload["comments"] = journal.get("comments")
        payload["journal_context"] = journal
        payload["work_notes_policy"] = {
            "same_cr_only": True,
            "role": "chronological auxiliary evidence from this CR",
            "use_for_context": True,
            "do_not_treat_as_current_field_state": True,
            "do_not_use_as_decision_label": True,
            "distinguish_claims_from_verified_evidence": True,
        }
        payload["retrieved_memory"] = compact_memory(list(payload.get("retrieved_memory") or []), limit=8)
        payload["prior_findings"] = compact_findings(list(payload.get("prior_findings") or []), limit=18)
        payload["strictness"] = context.strictness.value
        historical = self._compact_historical_context(self._historical_context())
        if historical is not None:
            payload["historical_training_context"] = historical
            payload["historical_training_context_policy"] = {
                "records_are_reference_examples_not_current_CR_facts": True,
                "historical_outcomes_are_aggregate_context": True,
                "never_copy_a_historical_prediction_to_the_current_CR": True,
                "delta_validate_any_similar_pattern": True,
            }
        payload["field_selection"] = {
            "selected_field_count": len(selected_cr),
            "selected_fields": list(selected_cr.keys()),
            "selection_source": "explicit-cr-schema-plus-field-agent",
            "descriptive_fields_fixed": True,
            "signoff_disposition_aware": True,
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
            "technical_reasoning": "substantive technical assessment",
            "cab_reasoning": "plain-language CAB decision rationale",
            "cab_questions": "array of up to 4 useful CAB questions",
            "recommendations": "array of up to 4 concrete next actions",
            "self_critique": "array of concise answers used to challenge the conclusion",
        }
        payload["instruction"] = (
            "Return ONLY one valid JSON object. First map the fixed CR schema to applicable requirements and evidence, "
            "then reason about readiness. Use all supplied descriptive fields, including explicit missing/present state. "
            "Interpret signoff fields by their disposition (Yes, No, Not Applicable, Completed, etc.), not merely whether "
            "they are populated. Work Notes and Comments are chronological journal evidence from this SAME CR; use them "
            "when relevant, but never let a later journal entry silently overwrite current structured fields. If the legacy "
            "combined journal field is supplied, keep that limitation explicit. Historical context is reference material, "
            "not current-CR evidence; identify patterns and delta-check them. Never copy a historical prediction. UAT is "
            "contextual. Rollback must be a credible recovery mechanism. Never invent approvals, testing, evidence, history, "
            "or policy. Separate facts, inferences, uncertainties and contradictions, and challenge false-PASS risk."
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
        critique = None
        if self.mode == "dual":
            critique = self.model.generate(
                system=build_reasoning_system_prompt() + " You are an independent adversarial reviewer. Return valid JSON using the same output contract and revise the prediction when warranted.",
                user=json.dumps({
                    "original_context": payload,
                    "initial_reasoning": initial.text,
                    "critique_contract": {
                        "challenge_every_inference": True,
                        "never_upgrade_unknown_to_fact": True,
                        "never_invent_evidence": True,
                        "look_for_contradictions": True,
                        "look_for_false_pass_risk": True,
                    },
                }, ensure_ascii=False, default=str),
                temperature=0.0,
                response_format=_JSON_OBJECT_FORMAT,
                reasoning_effort=self.reasoning_effort,
            )
        memory_cr = context.llm_cr if isinstance(context.llm_cr, dict) and context.llm_cr else context.cr
        clean_memory_cr = compact_cr(sanitize_cr_for_reasoning(memory_cr))
        memory_view = tuple(
            {"id": m.memory_id, "kind": m.kind.value, "text": m.text, "metadata": m.metadata, "score": m.score}
            for m in self._memories(clean_memory_cr)
        )
        return ReasoningLoopResult(initial=initial, critique=critique, retrieved_memory=memory_view, questions=self_critique_questions(), mode=self.mode)
