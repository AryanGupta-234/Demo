"""Deterministic similarity and clone/delta analysis contracts for Normal CRs."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class SimilarChange:
    change_id: str
    similarity: float
    historical_decision: str | None = None
    cab_recommendation: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class CloneAnalysis:
    candidate: SimilarChange
    reusable_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]
    revalidation_fields: tuple[str, ...]
    recommendation: str


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, (left or "").lower().strip(), (right or "").lower().strip()).ratio()


def change_similarity(current: dict[str, Any], historical: dict[str, Any]) -> float:
    """Weighted similarity focused on change semantics rather than CR identifiers."""
    weights = {
        "Type": 0.05,
        "Category": 0.15,
        "Sub Category": 0.10,
        "Short description": 0.25,
        "Description": 0.15,
        "Implementation plan": 0.15,
        "Configuration item": 0.10,
        "Company": 0.05,
    }
    total = 0.0
    score = 0.0
    for field, weight in weights.items():
        total += weight
        a = str(current.get(field) or "")
        b = str(historical.get(field) or "")
        if field in {"Type", "Category", "Sub Category", "Configuration item", "Company"}:
            part = 1.0 if a.strip().lower() == b.strip().lower() and a.strip() else 0.0
        else:
            part = text_similarity(a, b)
        score += weight * part
    return round(score / total, 4) if total else 0.0


def clone_analysis(current: dict[str, Any], historical: dict[str, Any], similarity: float | None = None) -> CloneAnalysis:
    sim = change_similarity(current, historical) if similarity is None else similarity
    candidate = SimilarChange(
        change_id=str(historical.get("Number") or historical.get("Effective number") or "unknown"),
        similarity=sim,
        historical_decision=str(historical.get("CAB Outcome") or historical.get("State") or "") or None,
        cab_recommendation=str(historical.get("CAB recommendation") or "") or None,
    )
    compare_fields = ("Category", "Sub Category", "Implementation plan", "Test plan", "Backout plan", "Customer Approval", "Risk", "Configuration item")
    reusable: list[str] = []
    changed: list[str] = []
    for field in compare_fields:
        a = str(current.get(field) or "").strip()
        b = str(historical.get(field) or "").strip()
        if text_similarity(a, b) >= 0.82:
            reusable.append(field)
        elif a or b:
            changed.append(field)
    revalidation = tuple(changed)
    if sim >= 0.90:
        recommendation = "CLONE_CANDIDATE"
    elif sim >= 0.75:
        recommendation = "RELATED_REFERENCE"
    else:
        recommendation = "DO_NOT_CLONE"
    return CloneAnalysis(candidate, tuple(reusable), tuple(changed), revalidation, recommendation)
