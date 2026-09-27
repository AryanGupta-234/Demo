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

        kind_weight = {
            MemoryKind.POLICY: 1.12,
            MemoryKind.EVIDENCE: 1.08,
            MemoryKind.FACT: 1.04,
            MemoryKind.CAB_HISTORY: 1.00,
            MemoryKind.SIMILARITY: 0.98,
            MemoryKind.EPISODE: 0.92,
        }

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
            haystack = f"{record.text} {record.metadata}".lower()
            if query_vector is not None:
                vector = self._vectors.get(record.memory_id)
                if vector is not None:
                    semantic = max(0.0, cosine_similarity(query_vector, vector))
            # Exact identifiers/components and semantic similarity are valuable,
            # but lexical overlap remains the strongest signal for CR governance.
            exact_identifier = 1.0 if any(token in haystack for token in terms if token.startswith(("chg", "req", "sr", "ci-"))) else 0.0
            base = 0.55 * lexical + 0.35 * semantic + 0.10 * exact_identifier
            return base * kind_weight.get(record.kind, 0.90)

        scored = [(combined_score(record), record) for record in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            MemoryRecord(record.memory_id, record.kind, record.text, record.metadata, round(score, 4))
            for score, record in scored[:limit]
            if score > 0
        ]


def remember_validation_episode(
    memory: UnifiedMemory,
    cr: dict[str, Any],
    *,
    decision: str,
    confidence: float | None = None,
    findings: list[Finding] | None = None,
    profile: dict[str, Any] | None = None,
    reasoning_summary: str = "",
) -> MemoryRecord:
    """Store a compact, non-authoritative episode from a validation run.

    Episodes are experience logs, not policy or ground truth. They help later
    runs retrieve similar failure patterns and reviewer context; the reasoning
    layer must delta-validate them against the current CR.
    """
    number = str(cr.get("Number") or cr.get("Effective number") or "UNKNOWN_CR").strip()
    finding_codes = [
        getattr(finding, "code", "")
        for finding in (findings or [])
        if getattr(finding, "severity", None) is not None
    ][:12]
    material = [
        getattr(finding, "message", "")
        for finding in (findings or [])
        if str(getattr(getattr(finding, "severity", None), "value", "")) in {"BLOCKING", "WARNING"}
    ][:6]
    archetype = str((profile or {}).get("primary_archetype") or "").strip()
    text_parts = [
        f"Validation episode for {number}: decision={decision}.",
        f"Change archetype={archetype or 'GENERAL'}.",
    ]
    if material:
        text_parts.append("Material observations: " + " | ".join(str(item) for item in material))
    if reasoning_summary:
        text_parts.append("Reasoning summary: " + reasoning_summary[:1800])

    metadata: dict[str, Any] = {
        "cr_number": number,
        "decision": decision,
        "learning_role": "case_history",
        "authoritative": False,
        "current_cr_evidence": False,
        "finding_codes": finding_codes,
        "profile": {
            "primary_archetype": archetype,
            "functional": bool((profile or {}).get("functional")),
            "infrastructure": bool((profile or {}).get("infrastructure")),
            "customer_facing": bool((profile or {}).get("customer_facing")),
        },
    }
    if confidence is not None:
        metadata["confidence"] = max(0.0, min(1.0, float(confidence)))

    import hashlib
    import time
    stamp = time.time_ns()
    digest = hashlib.sha1(f"{number}|{decision}|{stamp}".encode("utf-8")).hexdigest()[:12]
    record = MemoryRecord(
        memory_id=f"episode:{number}:{digest}",
        kind=MemoryKind.EPISODE,
        text="\n".join(text_parts),
        metadata=metadata,
    )
    memory.remember(record)
    return record
