"""Unified-memory abstractions used by all agents.

The initial implementation is storage-agnostic. A PostgreSQL/pgvector adapter can be added
without changing agent contracts. Memory records distinguish facts, policy, history and episodes.
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
    def search(self, query: str, *, kinds: list[MemoryKind] | None = None, limit: int = 10) -> list[MemoryRecord]:
        ...

    def remember(self, record: MemoryRecord) -> None:
        ...


class InMemoryUnifiedMemory:
    """Deterministic local memory for tests and the demo bootstrap."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def remember(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def search(self, query: str, *, kinds: list[MemoryKind] | None = None, limit: int = 10) -> list[MemoryRecord]:
        terms = {term.lower() for term in query.split() if term.strip()}
        candidates = self._records
        if kinds:
            candidates = [r for r in candidates if r.kind in kinds]

        def lexical_score(record: MemoryRecord) -> float:
            haystack = f"{record.text} {record.metadata}".lower()
            if not terms:
                return 0.0
            return sum(term in haystack for term in terms) / len(terms)

        ranked = sorted(candidates, key=lexical_score, reverse=True)
        return [MemoryRecord(r.memory_id, r.kind, r.text, r.metadata, lexical_score(r)) for r in ranked[:limit] if lexical_score(r) > 0]
