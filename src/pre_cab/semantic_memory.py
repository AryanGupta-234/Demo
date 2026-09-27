"""Persistent semantic memory engine.

Keeps the existing SQLite memory contract but adds a durable local vector index.
FTS remains a compatibility fallback; semantic similarity is the primary recall
signal when a local embedding provider is available.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .memory import MemoryKind, MemoryRecord
from .persistent_memory import SQLiteUnifiedMemory

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*", re.IGNORECASE)
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "change", "normal",
    "cr", "plan", "test", "testing", "please", "field", "current", "was", "are",
    "has", "have", "into", "then", "than", "but", "not", "only",
}


def _tokens(text: str) -> set[str]:
    return {
        x.lower() for x in _TOKEN_RE.findall(text or "")
        if len(x) >= 3 and x.lower() not in _STOPWORDS
    }


def _ids(text: str) -> set[str]:
    return {
        x.upper() for x in _TOKEN_RE.findall(text or "")
        if re.fullmatch(r"(?:CHG|REQ|SR)\d+|CI-[A-Z0-9._-]+", x, re.I)
    }


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dimension: int) -> list[float]:
    if not blob or dimension <= 0 or len(blob) != dimension * 4:
        return []
    return list(struct.unpack(f"<{dimension}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class SemanticSQLiteUnifiedMemory:
    """Unified persistent memory with local semantic recall and deterministic fallback."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: Any | None = None,
        semantic_weight: float = 0.65,
    ) -> None:
        self.path = str(path)
        self._lexical = SQLiteUnifiedMemory(path)
        self.embedding_provider = embedding_provider
        self.semantic_weight = max(0.0, min(1.0, float(semantic_weight)))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_vector_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_vector_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    text_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_model ON memory_vectors(model_name)")

    @staticmethod
    def _hash(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_provider or not texts:
            return []
        try:
            return self.embedding_provider.embed(texts)
        except Exception:
            return []

    def _upsert_vector(self, memory_id: str, text: str, vector: list[float]) -> None:
        if not vector:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_vectors(memory_id, model_name, dimension, vector, text_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model_name=excluded.model_name,
                    dimension=excluded.dimension,
                    vector=excluded.vector,
                    text_hash=excluded.text_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    str(getattr(self.embedding_provider, "model_name", "local")),
                    len(vector),
                    _pack(vector),
                    self._hash(text),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def remember(self, record: MemoryRecord) -> None:
        self._lexical.remember(record)
        vectors = self._embed([record.text])
        if vectors:
            self._upsert_vector(record.memory_id, record.text, vectors[0])

    def _backfill(self, rows: list[tuple[Any, ...]], limit: int = 128) -> None:
        if not self.embedding_provider:
            return
        missing = []
        for memory_id, text, metadata_json in rows[:limit]:
            with self._connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM memory_vectors WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
            if not exists:
                missing.append((memory_id, text))
        if not missing:
            return
        vectors = self._embed([text for _, text in missing])
        for (memory_id, text), vector in zip(missing, vectors):
            self._upsert_vector(memory_id, text, vector)

    def _load_rows(self, kinds: list[MemoryKind] | None) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                return conn.execute(
                    f"SELECT memory_id, kind, text, metadata_json FROM memories WHERE kind IN ({placeholders})",
                    tuple(k.value for k in kinds),
                ).fetchall()
            return conn.execute(
                "SELECT memory_id, kind, text, metadata_json FROM memories"
            ).fetchall()

    def _vector_map(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT memory_id, dimension, vector FROM memory_vectors WHERE memory_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {memory_id: _unpack(blob, int(dimension)) for memory_id, dimension, blob in rows}

    @staticmethod
    def _metadata_score(metadata: dict[str, Any]) -> float:
        confidence = metadata.get("confidence", metadata.get("source_confidence", 0.5))
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        authority = 1.08 if metadata.get("authoritative") else 1.0
        current = 1.08 if metadata.get("current_cr_evidence") else 1.0
        historical = 0.94 if metadata.get("historical_outcome") else 1.0
        return max(0.75, min(1.22, confidence * authority * current * historical))

    @staticmethod
    def _recency_score(metadata: dict[str, Any]) -> float:
        raw = str(metadata.get("_memory_updated_at") or metadata.get("_memory_created_at") or "")
        if not raw:
            return 0.5
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
            age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
            return max(0.55, 1.0 - min(0.45, age_days / 730.0))
        except ValueError:
            return 0.5

    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []

        rows = self._load_rows(kinds)
        if not rows:
            return []

        # Populate missing vectors incrementally, so upgrading an existing DB is online.
        self._backfill([(row[0], row[2], row[3]) for row in rows], limit=256)
        vectors = self._vector_map([row[0] for row in rows])
        query_vectors = self._embed([query])
        query_vector = query_vectors[0] if query_vectors else None

        q_tokens = _tokens(query)
        q_ids = _ids(query)
        scored: list[tuple[float, MemoryRecord, list[float]]] = []

        for memory_id, kind, text, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            haystack = f"{text} {metadata}"
            lexical = len(q_tokens & _tokens(haystack)) / max(1, len(q_tokens))
            identifier = 1.0 if q_ids and q_ids.intersection(_ids(haystack)) else 0.0
            vector = vectors.get(memory_id)
            semantic = max(0.0, _cosine(query_vector, vector)) if query_vector and vector else 0.0

            # Semantic similarity is primary, while exact IDs and governance authority
            # remain hard-to-ignore signals for change-management records.
            base = (
                self.semantic_weight * semantic
                + (0.18 - 0.08 * self.semantic_weight) * lexical
                + 0.12 * identifier
                + 0.08 * self._metadata_score(metadata)
                + 0.05 * self._recency_score(metadata)
            )
            if not query_vector or not vector:
                # Graceful compatibility mode for environments without sentence-transformers.
                base = 0.60 * lexical + 0.18 * identifier + 0.12 * self._metadata_score(metadata) + 0.10 * self._recency_score(metadata)

            if base <= 0:
                continue
            record = MemoryRecord(
                memory_id=memory_id,
                kind=MemoryKind(kind),
                text=text,
                metadata=metadata,
                score=round(base, 4),
            )
            scored.append((base, record, vector or []))

        scored.sort(key=lambda item: item[0], reverse=True)

        # MMR prevents the model context from being filled with near-duplicate episodes.
        selected: list[tuple[float, MemoryRecord, list[float]]] = []
        lambda_mult = 0.84
        while scored and len(selected) < limit:
            best_index = 0
            best_value = float("-inf")
            for index, item in enumerate(scored):
                relevance, record, vector = item
                redundancy = 0.0
                if vector:
                    redundancy = max(
                        (_cosine(vector, chosen[2]) for chosen in selected if chosen[2]),
                        default=0.0,
                    )
                mmr = lambda_mult * relevance - (1.0 - lambda_mult) * max(0.0, redundancy)
                if mmr > best_value:
                    best_value = mmr
                    best_index = index
            selected.append(scored.pop(best_index))

        results = [item[1] for item in selected]
        # Reuse the existing access accounting so old tooling and audit behaviour stay intact.
        self._lexical._touch([record.memory_id for record in results])
        return results

    def apply_feedback(self, memory_id: str, *, reward: float, note: str = "") -> None:
        """Record reviewer feedback without mutating the original memory text."""
        reward = max(-1.0, min(1.0, float(reward)))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return
            try:
                metadata = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            count = int(metadata.get("feedback_count", 0) or 0)
            total = float(metadata.get("feedback_reward", 0.0) or 0.0)
            metadata["feedback_count"] = count + 1
            metadata["feedback_reward"] = total + reward
            metadata["feedback_mean"] = (total + reward) / (count + 1)
            if note:
                metadata["feedback_notes"] = [*list(metadata.get("feedback_notes") or [])[-4:], note[:500]]
            conn.execute(
                "UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
                (json.dumps(metadata, default=str), memory_id),
            )

    def consolidate(self, *, max_records: int = 96, threshold: float = 0.88) -> int:
        """Create compact semantic pattern memories from repeated episodic experience."""
        if not self.embedding_provider:
            return 0
        rows = self._load_rows([MemoryKind.EPISODE])
        rows = rows[-max_records:]
        if len(rows) < 2:
            return 0
        self._backfill([(row[0], row[2], row[3]) for row in rows], limit=max_records)
        vectors = self._vector_map([row[0] for row in rows])
        clusters: list[list[tuple[Any, ...]]] = []
        for row in rows:
            vector = vectors.get(row[0])
            if not vector:
                continue
            placed = False
            for cluster in clusters:
                anchor = vectors.get(cluster[0][0], [])
                if _cosine(vector, anchor) >= threshold:
                    cluster.append(row)
                    placed = True
                    break
            if not placed:
                clusters.append([row])

        created = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            ids = [str(row[0]) for row in cluster]
            digest = self._hash("|".join(sorted(ids)))[:16]
            memory_id = f"pattern:{digest}"
            if any(self._lexical.search(memory_id, kinds=[MemoryKind.SIMILARITY], limit=1)):
                continue
            snippets = []
            for row in cluster[:5]:
                text = str(row[2]).replace("\n", " ")
                snippets.append(text[:420])
            text = (
                f"Consolidated experience pattern from {len(cluster)} related validation episodes. "
                + " | ".join(snippets)
            )
            self.remember(
                MemoryRecord(
                    memory_id=memory_id,
                    kind=MemoryKind.SIMILARITY,
                    text=text[:2400],
                    metadata={
                        "learning_role": "consolidated_pattern",
                        "authoritative": False,
                        "current_cr_evidence": False,
                        "derived_from": ids,
                        "pattern_strength": min(1.0, 0.55 + 0.08 * len(cluster)),
                        "consolidated_at": datetime.now(UTC).isoformat(),
                    },
                )
            )
            created += 1
        return created
