"""CAB-friendly and technical report rendering.

The same findings are presented at two levels: a concise CAB view and a deep technical view.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import Finding, FindingSeverity, ValidationResult


def _finding_dict(finding: Finding) -> dict[str, Any]:
    return {
        "code": finding.code,
        "title": finding.title,
        "severity": finding.severity.value,
        "message": finding.message,
        "technical_detail": finding.technical_detail,
        "evidence_refs": list(finding.evidence_refs),
        "recommendation": finding.recommendation,
    }


def build_report(result: ValidationResult, *, stage2: Any = None) -> dict[str, Any]:
    """Build a stable UI/API payload without hiding technical findings."""
    blockers = [f for f in result.findings if f.severity == FindingSeverity.BLOCKING]
    warnings = [f for f in result.findings if f.severity == FindingSeverity.WARNING]
    info = [f for f in result.findings if f.severity == FindingSeverity.INFO]

    cab_attention = [
        {
            "title": f.title,
            "message": f.message,
            "action": f.recommendation,
        }
        for f in (*blockers, *warnings)
    ]

    evidence_summary: dict[str, Any] = {}
    if stage2 is not None:
        evidence_summary = {
            "decision": stage2.decision.value,
            "confidence": stage2.confidence,
            "verified": stage2.verified,
            "contradictions": stage2.contradictions,
            "documents_considered": stage2.documents_considered,
        }

    return {
        "decision": result.decision.value,
        "confidence": round(result.confidence, 3),
        "score": round(result.score, 1),
        "strictness": result.strictness.value,
        "cab_view": {
            "summary": result.cab_summary,
            "attention_items": cab_attention,
            "requirements": [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in result.requirements],
            "evidence": evidence_summary,
        },
        "technical_view": {
            "summary": result.technical_summary,
            "blocking_findings": [_finding_dict(f) for f in blockers],
            "warnings": [_finding_dict(f) for f in warnings],
            "informational": [_finding_dict(f) for f in info],
            "historical_matches": result.historical_matches,
            "metadata": result.metadata,
        },
    }
