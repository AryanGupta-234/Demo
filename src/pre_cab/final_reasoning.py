"""Evidence-aware final reasoning and conservative decision reconciliation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .brain_loop import AgenticReasoningLoop, ReasoningLoopResult
from .evidence import EvidenceDocument, EvidenceResult
from .evidence_retrieval import retrieve_evidence_chunks
from .memory import UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Decision, Finding, FindingSeverity, Strictness, ValidationResult

_DECISION_RANK = {Decision.PASS: 0, Decision.CONDITIONAL: 1, Decision.NOT_READY: 2}
_REQUIRED_KEYS = {
    "prediction", "confidence", "facts", "inferences", "uncertainties", "contradictions",
    "technical_reasoning", "cab_reasoning", "cab_questions", "recommendations", "self_critique",
}


@dataclass(frozen=True)
class FinalReasoningResult:
    reasoning: ReasoningLoopResult | None
    model_prediction: Decision | None
    payload: dict[str, Any] | None
    error: str | None


def _parse(text: str) -> tuple[Decision | None, dict[str, Any] | None]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(payload, dict) or not _REQUIRED_KEYS.issubset(payload):
        return None, None
    try:
        decision = Decision(str(payload.get("prediction")))
    except ValueError:
        decision = None
    return decision, payload


def _evidence_context(
    cr: dict[str, Any],
    stage2: EvidenceResult | None,
    documents: list[EvidenceDocument] | None,
) -> list[dict[str, Any]]:
    if stage2 is None:
        return []
    chunks = retrieve_evidence_chunks(cr, documents or [], limit=12)
    return [{
        "decision": stage2.decision.value,
        "confidence": stage2.confidence,
        "verified": stage2.verified,
        "contradictions": list(stage2.contradictions),
        "documents_considered": list(stage2.documents_considered),
        "findings": [
            {
                "code": f.code,
                "severity": f.severity.value,
                "message": f.message,
                "technical_detail": f.technical_detail,
                "evidence_refs": list(f.evidence_refs),
            }
            for f in stage2.findings
        ],
        "ranked_attachment_excerpts": [
            {
                "document_ref": c.document_ref,
                "document_name": c.document_name,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "matched_terms": list(c.matched_terms),
                "text": c.text,
            }
            for c in chunks
        ],
    }]


def _model_has_material_signal(reasoning: FinalReasoningResult) -> bool:
    if not reasoning.payload:
        return False
    payload = reasoning.payload
    contradictions = payload.get("contradictions") or []
    technical = str(payload.get("technical_reasoning") or "").lower()
    return bool(contradictions) or any(
        term in technical
        for term in (
            "rollback", "recovery", "outage", "downtime", "dependency",
            "security exposure", "test execution", "failed", "failure",
            "contradiction", "approval", "implementation gap", "impact mismatch",
            "configuration drift", "evidence mismatch", "document mismatch",
        )
    )


def reconcile_final_decision(
    deterministic: Decision,
    reasoning: FinalReasoningResult,
) -> tuple[Decision, Finding | None]:
    prediction = reasoning.model_prediction
    if reasoning.error:
        return deterministic, Finding(
            code="FINAL_REASONING_UNAVAILABLE",
            title="Supplementary reasoning unavailable",
            severity=FindingSeverity.INFO,
            message="Primary validation completed, but the optional reasoning pass was unavailable.",
            technical_detail=reasoning.error,
            recommendation="Retry the reasoning pass or complete human review.",
        )
    if reasoning.reasoning is not None and reasoning.payload is None:
        return deterministic, Finding(
            code="FINAL_REASONING_INVALID",
            title="Supplementary reasoning response unusable",
            severity=FindingSeverity.INFO,
            message="The reasoning model responded, but its structured contract could not be validated.",
            recommendation="Retry the reasoning pass or complete human review.",
        )
    if prediction is None or _DECISION_RANK[prediction] <= _DECISION_RANK[deterministic]:
        return deterministic, None
    if not _model_has_material_signal(reasoning):
        return deterministic, None

    payload = reasoning.payload or {}
    recommendations = payload.get("recommendations") or []
    rec_text = (
        "; ".join(str(item) for item in recommendations[:3])
        if isinstance(recommendations, list)
        else str(recommendations)
    )
    return prediction, Finding(
        code="FINAL_MODEL_DOWNGRADE",
        title="Reasoning model identified additional evidence-aware CAB risk",
        severity=FindingSeverity.BLOCKING if prediction == Decision.NOT_READY else FindingSeverity.WARNING,
        message=str(payload.get("cab_reasoning") or "The final reasoning pass identified additional readiness risk."),
        technical_detail=str(payload.get("technical_reasoning") or ""),
        recommendation=rec_text,
    )


def run_final_reasoning(
    cr: dict[str, Any],
    validation: ValidationResult,
    stage2: EvidenceResult | None,
    *,
    strictness: Strictness,
    model: ModelProvider | None,
    memory: UnifiedMemory | None = None,
    documents: list[EvidenceDocument] | None = None,
) -> FinalReasoningResult:
    """Run one final reasoning pass after CR and evidence gates."""
    if model is None:
        return FinalReasoningResult(None, None, None, None)

    context = AgentContext(
        cr=cr,
        strictness=strictness,
        evidence=_evidence_context(cr, stage2, documents),
        prior_findings=tuple(validation.findings + (stage2.findings if stage2 else [])),
    )
    try:
        reasoning = AgenticReasoningLoop(model=model, memory=memory).run(
            context,
            findings=list(context.prior_findings),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return FinalReasoningResult(None, None, None, f"{type(exc).__name__}: {exc}")

    response = reasoning.critique or reasoning.initial
    prediction, payload = _parse(response.text)
    return FinalReasoningResult(reasoning, prediction, payload, None)
