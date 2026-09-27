"""Adaptive local reasoning loop for the Pre-CAB validator."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brain import build_narrative_system_prompt, build_reasoning_payload, build_reasoning_system_prompt, self_critique_questions
from .context_compaction import compact_cr, compact_evidence, compact_findings, compact_memory
from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider, ModelResponse
from .retrieval import build_cr_query
from .sanitization import sanitize_cr_for_reasoning, sanitize_value
from .rules import change_profile, contradiction_signals, effective_environment, evidence_matrix, model_field_validation, non_prod_validation_state
from .schemas import AgentContext, Finding

_REASONING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prediction": {"type": "string", "enum": ["PASS", "CONDITIONAL", "NOT_READY"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "facts": {"type": "array", "items": {"type": "string"}},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "technical_reasoning": {"type": "string"},
        "cab_reasoning": {"type": "string"},
        "cab_questions": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "self_critique": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "prediction", "confidence", "facts", "inferences", "uncertainties",
        "contradictions", "technical_reasoning", "cab_reasoning", "cab_questions",
        "recommendations", "self_critique",
    ],
}


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
    mode: str = "adaptive"
    final: ModelResponse | None = None


class AgenticReasoningLoop:
    """Reason over the mapped CR, its own Work Notes/Comments, requirements and learned history."""

    def __init__(self, model: ModelProvider, memory: UnifiedMemory | None = None, limit: int = 12, mode: str | None = None) -> None:
        self.model = model
        self.memory = memory
        self.limit = limit
        self.mode = (mode or os.getenv("PRE_CAB_REASONING_MODE") or "adaptive").strip().lower()
        if self.mode not in {"single", "dual", "adaptive"}:
            raise ValueError("PRE_CAB_REASONING_MODE must be 'single', 'dual', or 'adaptive'")
        effort = (os.getenv("PRE_CAB_REASONING_EFFORT") or "medium").strip().lower()
        if effort not in {"low", "medium", "high"}:
            effort = "medium"
        self.reasoning_effort = effort

    def _memory_queries(self, cr: dict[str, Any]) -> list[tuple[str, list[MemoryKind]]]:
        """Decompose recall into independent lanes before fusion."""
        core = " ".join(str(cr.get(k) or "") for k in (
            "Short description", "Description", "Category", "Sub Category", "Change Class"
        ))
        technical = " ".join(str(cr.get(k) or "") for k in (
            "Configuration item", "Implementation plan", "Change plan",
            "Backout plan", "Risk and impact analysis"
        ))
        testing = " ".join(str(cr.get(k) or "") for k in (
            "Test plan", "UAT signoff", "Test Results Evidence",
            "Lower Environment Reference CR/SR", "Work notes", "Comments"
        ))
        governance = " ".join(str(cr.get(k) or "") for k in (
            "Customer Approval", "TCS QA signoff", "Risk", "Priority",
            "Conflict status"
        ))
        return [
            (core or build_cr_query(cr), [MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY, MemoryKind.FACT]),
            (technical, [MemoryKind.CAB_HISTORY, MemoryKind.POLICY, MemoryKind.EVIDENCE, MemoryKind.SIMILARITY]),
            (testing, [MemoryKind.EVIDENCE, MemoryKind.CAB_HISTORY, MemoryKind.EPISODE]),
            (governance, [MemoryKind.POLICY, MemoryKind.CAB_HISTORY, MemoryKind.EPISODE]),
        ]

    def _memories(self, cr: dict[str, Any]) -> list[Any]:
        if not self.memory:
            return []
        candidates: dict[str, Any] = {}
        lane_count: dict[str, int] = {}
        for query, kinds in self._memory_queries(cr):
            query = " ".join(query.split()).strip()
            if not query:
                continue
            for item in self.memory.search(query, kinds=kinds, limit=max(4, self.limit // 2)):
                candidates[item.memory_id] = item
                lane_count[item.memory_id] = lane_count.get(item.memory_id, 0) + 1

        ranked: list[tuple[float, int, Any]] = []
        for item in candidates.values():
            score = float(item.score or 0.0)
            score = min(1.0, score + min(0.12, 0.04 * (lane_count.get(item.memory_id, 1) - 1)))
            ranked.append((score, lane_count.get(item.memory_id, 1), item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

        selected: list[Any] = []
        used_kinds: set[MemoryKind] = set()
        for _, _, item in ranked:
            if item.kind not in used_kinds:
                selected.append(item)
                used_kinds.add(item.kind)
            if len(selected) >= min(self.limit, len(ranked)):
                break
        if len(selected) < min(self.limit, len(ranked)):
            for _, _, item in ranked:
                if item in selected:
                    continue
                selected.append(item)
                if len(selected) >= self.limit:
                    break
        return selected[:self.limit]

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
    def _load_json_artifact(env_name: str) -> dict[str, Any] | None:
        raw = os.getenv(env_name, "").strip()
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

    @classmethod
    def _historical_context(cls) -> dict[str, Any] | None:
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
    def _learned_knowledge() -> dict[str, Any] | None:
        return AgenticReasoningLoop._load_json_artifact("PRE_CAB_LEARNED_KNOWLEDGE")

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
        payload["model_field_validation"] = model_field_validation(context.cr)
        payload["change_profile"] = change_profile(context.cr)
        payload["evidence_matrix"] = evidence_matrix(context.cr)
        payload["deterministic_contradictions"] = contradiction_signals(context.cr)
        payload["workflow_context"] = {
            "effective_environment": effective_environment(context.cr),
            "environment_is_derived_when_unset": not bool(str(context.cr.get("Environment") or "").strip()),
            "change_workflow_target": "PROD",
            "non_prod_validation": non_prod_validation_state(context.cr),
        }
        payload["evidence_interpretation_policy"] = {
            "classify_before_scoring": True,
            "current_execution_beats_plan": True,
            "historical_references_are_context_only": True,
            "formal_test_results_not_universal_for_infrastructure": True,
            "non_prod_labels_equivalent": ["SIT", "UAT", "Pre-PROD", "staging", "lower environment", "test environment"],
            "missing_metadata_is_not_automatically_a_technical_failure": True,
        }
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
        payload["agent_blackboard"] = {
            name: {
                "findings": list((state or {}).get("findings", []))[:8],
                "requirements": list((state or {}).get("requirements", []))[:8],
                "notes": self._compact_agent_notes((state or {}).get("notes", {})),
            }
            for name, state in context.agent_state.items()
            if isinstance(state, dict)
        }

        learned = self._learned_knowledge()
        if learned is not None:
            knowledge = learned.get("knowledge") if isinstance(learned.get("knowledge"), dict) else learned
            payload["learned_organization_knowledge"] = knowledge
            payload["learned_knowledge_policy"] = {
                "source": "historical_learning_stage",
                "status": "prior_knowledge_only",
                "historical_labels_used_to_build_it": True,
                "do_not_treat_as_current_cr_evidence": True,
                "do_not_copy_a_historical_outcome": True,
                "delta_validate_against_current_cr": True,
            }

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
            "technical_reasoning": (
                "substantive technical assessment for an engineer: name the specific fields/evidence checked, "
                "what they showed, and why that does or does not support readiness. Distinguish implementation, "
                "rollback, CI traceability, testing execution, formal evidence, dependencies and environment context."
            ),
            "cab_reasoning": (
                "plain-language CAB decision rationale for a non-technical reviewer: explain business/operational "
                "impact, what evidence supports the change, and what genuinely needs attention. Do not present "
                "a derived PROD environment, contextual N/A, or CI traceability observation as a missing control."
            ),
            "cab_questions": (
                "array of up to 4 useful CAB questions. Ask only about genuine unresolved evidence, execution, "
                "rollback, dependencies, impact or contradictions; do not ask for UAT/pre-PROD merely because "
                "those fields exist, especially for infrastructure changes."
            ),
            "recommendations": "array of up to 4 concrete next actions",
            "self_critique": (
                "array of concise answers to the self_critique_questions above — actually answer each "
                "one for this CR, do not repeat the questions themselves back verbatim."
            ),
            "nlg_quality": {
                "cab_reasoning": "2-4 natural sentences, evidence-first, readable without technical jargon, no generic filler, no repeated conclusion.",
                "technical_reasoning": "4-8 precise sentences using field/evidence names and causal reasoning; distinguish observed facts from inference.",
                "facts": "short evidence-grounded statements; never contain recommendations.",
                "inferences": "explicitly derived from facts and applicable requirements.",
                "uncertainties": "only unresolved or unverifiable items.",
                "contradictions": "only conflicts supported by two or more current signals.",
                "recommendations": "specific next actions, not generic review language.",
            },
        }
        payload["instruction"] = (
            "Return ONLY one valid JSON object. Analyze the current CR first using the fixed schema, mapped requirements, "
            "Work Notes, Comments, retrieved evidence and learned organization knowledge. The learned knowledge was derived "
            "from historical CRs and is prior knowledge, not current-CR evidence. Historical labels were used only during the "
            "learning pass and must not be copied to the current CR. Use all descriptive fields, including explicit missing/"
            "present state. Interpret signoff fields by disposition (Yes, No, Not Applicable, Completed, etc.). Work Notes and "
            "Comments are chronological journal evidence from this SAME CR. Never let a later journal entry silently overwrite "
            "current structured fields. UAT is contextual. Rollback must be credible. Never invent approvals, testing, evidence, "
            "history, or policy. Separate facts, inferences, uncertainties and contradictions and challenge false-PASS risk. "
            "Treat SIT/UAT/Pre-PROD/lower-environment references as one non-PROD validation class, do not report an unset "
            "Environment as missing in this PROD workflow, and keep missing CI as traceability unless explicitly gated. "
            "Validate all 14 model-facing fields even when some are legitimately unset or Not Applicable. Classify the change before deciding "
            "which evidence is mandatory. Use change_profile and evidence_matrix as deterministic context. Do not turn informational "
            "governance gaps into readiness blockers. A historical CR reference proves only that an example exists, not that this CR "
            "was tested. cab_reasoning and technical_reasoning serve two different readers and are shown to both together — write "
            "cab_reasoning so a non-technical CAB member understands the decision and its business impact on its own, and "
            "write technical_reasoning so an engineer gets the specific field-level evidence behind it; do not make one "
            "depend on the other to be understood. Use varied natural language, avoid repeating the same sentence or conclusion, "
            "lead each rationale with observed evidence, and never use empty phrases such as 'further review is recommended' "
            "unless the response names exactly what must be reviewed."
        )
        return payload

    @staticmethod
    def _compact_agent_notes(notes: Any) -> dict[str, Any]:
        if not isinstance(notes, dict):
            return {}
        compact: dict[str, Any] = {}
        for key, value in notes.items():
            if key in {"selected_cr", "matches", "clone_candidates"}:
                continue
            if isinstance(value, str):
                compact[key] = value[:1200]
            elif isinstance(value, list):
                compact[key] = value[:8]
            elif isinstance(value, dict):
                compact[key] = {str(k): v for k, v in list(value.items())[:20]}
            elif isinstance(value, (bool, int, float)) or value is None:
                compact[key] = value
        return compact

    @staticmethod
    def _parse_initial(text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        required = set(_REASONING_SCHEMA.get("required", ()))
        if not required.issubset(value):
            return None
        if str(value.get("prediction", "")).upper() not in {"PASS", "CONDITIONAL", "NOT_READY"}:
            return None
        try:
            confidence = float(value.get("confidence"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        cab_reasoning = str(value.get("cab_reasoning") or "").strip()
        technical_reasoning = str(value.get("technical_reasoning") or "").strip()
        if len(cab_reasoning) < 45 or len(technical_reasoning) < 90:
            return None
        if cab_reasoning == technical_reasoning:
            return None
        recommendations = value.get("recommendations")
        if not isinstance(recommendations, list) or len(recommendations) > 4:
            return None
        return value

    def _should_critique(self, response: ModelResponse) -> bool:
        if self.mode == "dual":
            return True
        if self.mode == "single":
            return False
        payload = self._parse_initial(response.text)
        if not payload:
            return True
        prediction = str(payload.get("prediction", "")).upper()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return (
            prediction != "PASS"
            or bool(payload.get("contradictions"))
            or bool(payload.get("uncertainties"))
            or confidence < 0.88
        )

    def _generate(self, *, system: str, user: str, temperature: float = 0.03) -> ModelResponse:
        response = self.model.generate(
            system=system,
            user=user,
            temperature=temperature,
            response_format=_REASONING_SCHEMA,
            reasoning_effort=self.reasoning_effort,
        )
        if self._parse_initial(response.text) is not None:
            return response

        # One bounded repair pass prevents malformed small-model output from
        # poisoning the rest of the decision chain.
        repair_user = json.dumps({
            "invalid_response": response.text[:12000],
            "instruction": "Return the same analysis as ONE valid JSON object matching the supplied schema. Do not add commentary outside JSON.",
        }, ensure_ascii=False)
        repaired = self.model.generate(
            system=system + " Your previous response failed schema validation. Repair it without inventing new evidence.",
            user=repair_user,
            temperature=0.0,
            response_format=_REASONING_SCHEMA,
            reasoning_effort=self.reasoning_effort,
        )
        return repaired

    def run(self, context: AgentContext, findings: list[Finding] | None = None) -> ReasoningLoopResult:
        payload = self._build_payload(context, findings)
        initial = self._generate(
            system=build_reasoning_system_prompt(),
            user=json.dumps(payload, ensure_ascii=False, default=str),
        )
        critique = None
        if self._should_critique(initial):
            critique = self._generate(
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
            )
        final_response = critique or initial
        nlg_mode = (os.getenv("PRE_CAB_NLG_MODE") or "refine").strip().lower()
        if nlg_mode not in {"off", "refine"}:
            nlg_mode = "refine"

        # Keep reasoning and language realization separate. The second call may improve
        # readability, but its decision/confidence are hard-locked to the reasoning pass.
        if nlg_mode == "refine":
            base_response = final_response
            parsed = self._parse_initial(base_response.text)
            if isinstance(parsed, dict):
                narrative_input = {
                    "locked_decision": parsed.get("prediction"),
                    "locked_confidence": parsed.get("confidence"),
                    "facts": parsed.get("facts", []),
                    "inferences": parsed.get("inferences", []),
                    "uncertainties": parsed.get("uncertainties", []),
                    "contradictions": parsed.get("contradictions", []),
                    "technical_reasoning": parsed.get("technical_reasoning", ""),
                    "cab_reasoning": parsed.get("cab_reasoning", ""),
                    "cab_questions": parsed.get("cab_questions", []),
                    "recommendations": parsed.get("recommendations", []),
                    "self_critique": parsed.get("self_critique", []),
                }
                try:
                    nlg = self.model.generate(
                        system=build_narrative_system_prompt(),
                        user=json.dumps(narrative_input, ensure_ascii=False, default=str),
                        temperature=0.18,
                        response_format=_REASONING_SCHEMA,
                        reasoning_effort="low",
                    )
                    narrative = self._parse_initial(nlg.text)
                except Exception:
                    narrative = None

                if isinstance(narrative, dict):
                    merged = dict(parsed)
                    for key in ("technical_reasoning", "cab_reasoning", "cab_questions", "recommendations", "self_critique"):
                        if key in narrative:
                            merged[key] = narrative[key]
                    merged["prediction"] = parsed.get("prediction")
                    merged["confidence"] = parsed.get("confidence")
                    final_response = ModelResponse(
                        text=json.dumps(merged, ensure_ascii=False, default=str),
                        model=base_response.model,
                        raw={"reasoning": base_response.raw, "narrative": nlg.raw},
                    )

        memory_cr = context.llm_cr if isinstance(context.llm_cr, dict) and context.llm_cr else context.cr
        clean_memory_cr = compact_cr(sanitize_cr_for_reasoning(memory_cr))
        memory_view = tuple(
            {"id": m.memory_id, "kind": m.kind.value, "text": m.text, "metadata": m.metadata, "score": m.score}
            for m in self._memories(clean_memory_cr)
        )
        return ReasoningLoopResult(
            initial=initial,
            critique=critique,
            retrieved_memory=memory_view,
            questions=self_critique_questions(),
            mode=self.mode,
            final=final_response,
        )
