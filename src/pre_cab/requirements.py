"""Compatibility layer over the centralized Normal-CR rule engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rules import context_flags, contextual_signals, field_quality, required_field_policies


@dataclass(frozen=True)
class RequirementPrediction:
    name: str
    required: bool
    confidence: float
    reason: str
    signals: tuple[str, ...] = ()


def infer_requirements(cr: dict[str, Any]) -> list[RequirementPrediction]:
    """Return contextual and field-level requirements from the single rule engine."""
    predictions: list[RequirementPrediction] = []

    for policy, required, matched_flags in required_field_policies(cr):
        if required:
            reason = (
                f"Baseline field for an in-scope Normal CR: {policy.label}."
                if policy.baseline
                else f"Required because contextual condition(s) matched: {', '.join(matched_flags)}."
            )
            predictions.append(
                RequirementPrediction(
                    name=policy.field,
                    required=True,
                    confidence=0.99 if policy.baseline else 0.88,
                    reason=reason,
                    signals=matched_flags or ("baseline",),
                )
            )

    for signal in contextual_signals(cr):
        predictions.append(
            RequirementPrediction(
                name=signal.requirement,
                required=signal.applicable,
                confidence=signal.confidence,
                reason=signal.rationale,
                signals=signal.required_fields + signal.evidence_fields,
            )
        )

    return predictions


def requirement_context(cr: dict[str, Any]) -> dict[str, Any]:
    """Expose explainable rule state to agents and the final report."""
    return {
        "context_flags": context_flags(cr),
        "required_fields": [policy.field for policy, required, _ in required_field_policies(cr) if required],
        "field_quality": [
            {"field": item.field, "present": item.present, "score": item.score, "reasons": list(item.reasons)}
            for item in field_quality(cr)
        ],
        "contextual_signals": [
            {
                "requirement": item.requirement,
                "applicable": item.applicable,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "source": item.source,
                "required_fields": list(item.required_fields),
                "evidence_fields": list(item.evidence_fields),
                "severity": item.severity,
            }
            for item in contextual_signals(cr)
        ],
    }
