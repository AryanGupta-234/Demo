"""Historical similarity and safe clone/delta analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimilarityMatch:
    cr_number: str
    similarity: float
    outcome: str | None
    reusable_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def similarity_score(current: dict[str, Any], historical: dict[str, Any]) -> float:
    """Cheap deterministic baseline; embeddings/reranking can replace this later."""
    weighted = {
        "Category": 0.15,
        "Sub Category": 0.10,
        "Configuration item": 0.15,
        "Short description": 0.25,
        "Description": 0.15,
        "Implementation plan": 0.10,
        "Backout plan": 0.05,
        "Type": 0.05,
    }
    score = 0.0
    for field, weight in weighted.items():
        a, b = _norm(current.get(field)), _norm(historical.get(field))
        if not a or not b:
            continue
        if a == b:
            score += weight
            continue
        a_words, b_words = set(a.split()), set(b.split())
        if a_words and b_words:
            score += weight * (len(a_words & b_words) / len(a_words | b_words))
    return round(score, 4)


def compare_for_clone(current: dict[str, Any], historical: dict[str, Any]) -> SimilarityMatch:
    reusable: list[str] = []
    changed: list[str] = []
    comparison_fields = (
        "Category", "Sub Category", "Configuration item", "Implementation plan",
        "Backout plan", "Test plan", "Risk", "Risk and impact analysis",
    )
    for field in comparison_fields:
        if _norm(current.get(field)) == _norm(historical.get(field)) and _norm(current.get(field)):
            reusable.append(field)
        else:
            changed.append(field)

    return SimilarityMatch(
        cr_number=str(historical.get("Number", "")),
        similarity=similarity_score(current, historical),
        outcome=historical.get("CAB recommendation") or historical.get("CAB Outcome"),
        reusable_fields=tuple(reusable),
        changed_fields=tuple(changed),
    )


def clone_candidates(
    current: dict[str, Any], historical_crs: list[dict[str, Any]], *, minimum: float = 0.75, limit: int = 5
) -> list[SimilarityMatch]:
    matches = [compare_for_clone(current, old) for old in historical_crs]
    matches = [m for m in matches if m.similarity >= minimum]
    return sorted(matches, key=lambda m: m.similarity, reverse=True)[:limit]
