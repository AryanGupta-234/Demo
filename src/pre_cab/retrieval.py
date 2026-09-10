"""Hybrid retrieval baseline: structured filters, lexical search, then semantic-style reranking.

The interface is storage-agnostic. PostgreSQL/pgvector/OpenSearch adapters can replace the in-memory
implementation without changing agent contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalItem:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    sources: tuple[str, ...] = ()


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().replace("/", " ").replace("-", " ").split() if len(t) > 2}


def lexical_overlap(query: str, text: str) -> float:
    q, t = _tokens(query), _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def semantic_baseline(query: str, text: str) -> float:
    return SequenceMatcher(None, query.lower().strip(), text.lower().strip()).ratio()


def hybrid_retrieve(
    query: str,
    records: Iterable[RetrievalItem],
    *,
    metadata_filters: dict[str, Any] | None = None,
    limit: int = 8,
) -> list[RetrievalItem]:
    """Retrieve candidates using metadata, lexical overlap and semantic similarity."""
    metadata_filters = {k: v for k, v in (metadata_filters or {}).items() if v not in (None, "")}
    candidates: list[RetrievalItem] = []
    for record in records:
        if metadata_filters and any(record.metadata.get(k) != v for k, v in metadata_filters.items()):
            continue
        lexical = lexical_overlap(query, record.text)
        semantic = semantic_baseline(query, record.text)
        score = round(0.60 * lexical + 0.40 * semantic, 4)
        if score > 0:
            candidates.append(
                RetrievalItem(record.id, record.text, record.metadata, score, tuple(dict.fromkeys(record.sources + ("hybrid",))))
            )
    return sorted(candidates, key=lambda x: x.score, reverse=True)[:limit]


def build_cr_query(cr: dict[str, Any]) -> str:
    fields = ("Short description", "Description", "Justification", "Category", "Sub Category", "Implementation plan")
    return " ".join(str(cr.get(field) or "") for field in fields)
