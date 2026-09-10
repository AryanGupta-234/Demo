"""Unified-memory abstractions used by all agents.

Memory is storage-agnostic and supports deterministic lexical retrieval by default. When an optional
embedding provider is supplied, search combines lexical and semantic similarity without changing the
agent contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class MemoryKind(str, Enum):
    FACT = "fact"
    POLICY = "policy"
    CAB_HISTORY = "cab_history"
    EVIDENCE = "evidence"
    EPISODE = "episode"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    kind: MemoryKind
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class UnifiedMemory(Protocol):
    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        ...

    def remember(self, record: MemoryRecord) -> None:
        ...


class InMemoryUnifiedMemory:
    """Deterministic memory with optional semantic ranking for the local/free-tier stack."""

    def __init__(self, embedding_provider: Any | None = None) -> None:
        self._records: list[MemoryRecord] = []
        self._embedding_provider = embedding_provider
        self._vectors: dict[str, list[float]] = {}

    def remember(self, record: MemoryRecord) -> None:
        self._records.append(record)
        if self._embedding_provider is not None:
            try:
                self._vectors[record.memory_id] = self._embedding_provider.embed([record.text])[0]
            except Exception:
                # Embeddings are an optimization. Never make core memory unavailable if embedding fails.
                pass

    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        terms = {term.lower() for term in query.split() if term.strip()}
        candidates = self._records
        if kinds:
            candidates = [record for record in candidates if record.kind in kinds]

        def lexical_score(record: MemoryRecord) -> float:
            haystack = f"{record.text} {record.metadata}".lower()
            if not terms:
                return 0.0
            return sum(term in haystack for term in terms) / len(terms)

        query_vector: list[float] | None = None
        if self._embedding_provider is not None:
            try:
                query_vector = self._embedding_provider.embed([query])[0]
            except Exception:
                query_vector = None

        from .embeddings import cosine_similarity

        def combined_score(record: MemoryRecord) -> float:
            lexical = lexical_score(record)
            semantic = 0.0
            if query_vector is not None:
                vector = self._vectors.get(record.memory_id)
                if vector is not None:
                    semantic = max(0.0, cosine_similarity(query_vector, vector))
            # Lexical precision is intentionally weighted more heavily for CR identifiers/components.
            return 0.65 * lexical + 0.35 * semantic

        scored = [(combined_score(record), record) for record in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            MemoryRecord(record.memory_id, record.kind, record.text, record.metadata, round(score, 4))
            for score, record in scored[:limit]
            if score > 0
        ]
