"""Evidence-aware final reasoning and conservative decision reconciliation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .brain_loop import AgenticReasoningLoop, ReasoningLoopResult
from .evidence import EvidenceResult
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
    except Exception:
        return None, None
    if not isinstance(payload, dict) or not _REQUIRED_KEYS.issubset(payload):
        return None, None
    try:
        decision = Decision(str(payload.get("prediction")))
    except ValueError:
        decision = None
    return decision, payload


def _evidence_context(stage2: EvidenceResult | None) -> list[dict[str, Any]]:
    if stage2 is None:
        return []
    return [
        {
            "decision": stage2.decision.value,
            "confidence": stage2.confidence,
            "verified": stage2.verified,
            "contradictions": list(stage2.contradictions),
            "documents_considered": list(stage2.documents_considered),
            "findings": [
                {
                    "code": finding.code,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "technical_detail": finding.technical_detail,
                    "evidence_refs": list(finding.evidence_refs),
                }
                for finding in stage2.findings
            ],
        }
    ]


def run_final_reasoning(
    cr: dict[str, Any],
    validation: ValidationResult,
    stage2: EvidenceResult | None,
    *,
    strictness: Strictness,
    model: ModelProvider | None,
    memory: UnifiedMemory | None = None,
) -> FinalReasoningResult:
    """Run the one high-value GPT pass after deterministic/evidence validation.

    GPT can only make readiness more conservative. Model failure/invalid output never upgrades a CR.
    """
    if model is None:
        return FinalReasoningResult(None, None, None, None)

    context = AgentContext(
        cr=cr,
        strictness=strictness,
        evidence=_evidence_context(stage2),
        prior_findings=tuple(validation.findings + (stage2.findings if stage2 else [])),
    )
    try:
        reasoning = AgenticReasoningLoop(model=model, memory=memory).run(
            context,
            findings=list(context.prior_findings),
        )
    except Exception as exc:
        return FinalReasoningResult(None, None, None, f"{type(exc).__name__}: {exc}")

    response = reasoning.critique or reasoning.initial
    prediction, payload = _parse(response.text)
    return FinalReasoningResult(reasoning, prediction, payload, None)


def reconcile_final_decision(
    deterministic: Decision,
    reasoning: FinalReasoningResult,
) -> tuple[Decision, Finding | None]:
    prediction = reasoning.model_prediction
    if reasoning.error:
        decision = Decision.CONDITIONAL if deterministic == Decision.PASS else deterministic
        return decision, Finding(
            code="FINAL_REASONING_UNAVAILABLE",
            title="Final GPT reasoning incomplete",
            severity=FindingSeverity.WARNING,
            message="Field/evidence validation completed, but the final GPT-OSS reasoning pass failed.",
            technical_detail=reasoning.error,
            recommendation="Retry the reasoning provider or complete human review.",
        )
    if reasoning.reasoning is not None and reasoning.payload is None:
        decision = Decision.CONDITIONAL if deterministic == Decision.PASS else deterministic
        return decision, Finding(
            code="FINAL_REASONING_INVALID",
            title="Final GPT reasoning output invalid",
            severity=FindingSeverity.WARNING,
            message="GPT-OSS responded, but the final structured assessment could not be validated.",
            recommendation="Retry the reasoning pass or complete human review.",
        )
    if prediction is None or _DECISION_RANK[prediction] <= _DECISION_RANK[deterministic]:
        return deterministic, None

    payload = reasoning.payload or {}
    recommendation = payload.get("recommendations") or []
    return prediction, Finding(
        code="FINAL_MODEL_DOWNGRADE",
        title="GPT-OSS identified additional evidence-aware CAB risk",
        severity=FindingSeverity.BLOCKING if prediction == Decision.NOT_READY else FindingSeverity.WARNING,
        message=str(payload.get("cab_reasoning") or "The final reasoning pass identified additional risk."),
        technical_detail=str(payload.get("technical_reasoning") or ""),
        recommendation="; ".join(str(item) for item in recommendation[:3]) if isinstance(recommendation, list) else str(recommendation),
    )
