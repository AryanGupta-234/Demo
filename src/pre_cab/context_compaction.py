"""Deterministic context compaction for bounded GPT-OSS prompts."""
from __future__ import annotations

from typing import Any


_HIGH_VALUE_CR_FIELDS = (
    "Number", "Type", "Short description", "Description", "Justification", "Category", "Sub Category",
    "Change Class", "Risk", "Risk and impact analysis", "Configuration item", "Production system",
    "Implementation plan", "Backout plan", "Test plan", "Test Results Evidence", "UAT signoff",
    "Customer Approval", "Conflict status", "Planned start date", "Planned end date", "Environment",
)

_FIELD_LIMITS = {
    "Description": 2400,
    "Justification": 1600,
    "Risk and impact analysis": 2000,
    "Implementation plan": 3200,
    "Backout plan": 2200,
    "Test plan": 2600,
}


def _truncate(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + f"\n[TRUNCATED {len(value) - limit} chars]"


def compact_cr(cr: dict[str, Any]) -> dict[str, Any]:
    """Keep decision-relevant CR fields and bound long free-text values."""
    result: dict[str, Any] = {}
    for field in _HIGH_VALUE_CR_FIELDS:
        if field not in cr:
            continue
        value = cr[field]
        result[field] = _truncate(value, _FIELD_LIMITS.get(field, 900))
    return result


def compact_memory(records: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records[:limit]:
        result.append(
            {
                "id": record.get("id"),
                "kind": record.get("kind"),
                "score": record.get("score"),
                "text": _truncate(str(record.get("text") or ""), 1400),
                "metadata": {
                    key: value
                    for key, value in dict(record.get("metadata") or {}).items()
                    if key in {"cr_number", "historical_outcome", "category", "sub_category", "change_class"}
                },
            }
        )
    return result


def compact_findings(findings: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    severity_rank = {"blocking": 0, "warning": 1, "info": 2}
    ordered = sorted(findings, key=lambda item: severity_rank.get(str(item.get("severity")), 3))
    compacted: list[dict[str, Any]] = []
    for finding in ordered[:limit]:
        compacted.append(
            {
                "code": finding.get("code"),
                "severity": finding.get("severity"),
                "message": _truncate(str(finding.get("message") or ""), 700),
                "technical_detail": _truncate(str(finding.get("technical_detail") or ""), 900),
            }
        )
    return compacted


def compact_evidence(evidence: list[Any], limit: int = 12) -> list[Any]:
    result: list[Any] = []
    for item in evidence[:limit]:
        if isinstance(item, dict):
            compacted = dict(item)
            if "findings" in compacted and isinstance(compacted["findings"], list):
                compacted["findings"] = compact_findings(compacted["findings"], limit=16)
            result.append(compacted)
        else:
            result.append(_truncate(str(item), 1600))
    return result
