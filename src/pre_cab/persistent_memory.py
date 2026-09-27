"""Persistent unified memory with FTS retrieval, recency/confidence signals and lightweight learning metadata.

The memory remains dependency-free SQLite. It stores facts, policy, historical CRs, evidence,
episodes and similarity examples behind one retrieval contract. Retrieval is reranked using
identifier/entity matches, lexical coverage, source confidence, memory kind and recency so the
model sees useful memories rather than a flat keyword dump.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .memory import MemoryKind, MemoryRecord


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*", re.IGNORECASE)
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "change", "normal",
    "cr", "plan", "test", "testing", "please", "field", "current",
}
_KIND_WEIGHT = {
    MemoryKind.POLICY.value: 1.15,
    MemoryKind.EVIDENCE.value: 1.10,
    MemoryKind.CAB_HISTORY.value: 1.00,
    MemoryKind.SIMILARITY.value: 0.98,
    MemoryKind.EPISODE.value: 0.92,
    MemoryKind.FACT.value: 1.05,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _identifier_tokens(text: str) -> set[str]:
    return {
        token.upper()
        for token in _TOKEN_RE.findall(text or "")
        if re.fullmatch(r"(?:CHG|REQ|SR)\d+|CI-[A-Z0-9._-]+", token, re.IGNORECASE)
    }


class SQLiteUnifiedMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(memory_id UNINDEXED, kind UNINDEXED, text, metadata_json)
                    """
                )
                self._fts_enabled = True
                existing = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                if existing == 0:
                    conn.execute(
                        "INSERT INTO memories_fts(memory_id, kind, text, metadata_json) "
                        "SELECT memory_id, kind, text, metadata_json FROM memories"
                    )
            except sqlite3.OperationalError:
                self._fts_enabled = False

    @staticmethod
    def _enrich_metadata(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
        existing = dict(existing or {})
        merged = dict(existing)
        merged.update(incoming)
        now = _now()
        created = str(existing.get("_memory_created_at") or now)
        hits = int(existing.get("_memory_hits", 0) or 0)
        merged["_memory_created_at"] = created
        merged["_memory_updated_at"] = now
        merged["_memory_hits"] = hits + 1
        merged["_memory_last_accessed"] = existing.get("_memory_last_accessed")
        merged["_memory_version"] = int(existing.get("_memory_version", 0) or 0) + 1
        return merged

    def remember(self, record: MemoryRecord) -> None:
        with self._connect() as conn:
            existing_row = conn.execute(
                "SELECT metadata_json FROM memories WHERE memory_id = ?",
                (record.memory_id,),
            ).fetchone()
            existing = json.loads(existing_row[0]) if existing_row else None
            metadata = self._enrich_metadata(existing, dict(record.metadata))

            conn.execute(
                """
                INSERT INTO memories(memory_id, kind, text, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind=excluded.kind,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json
                """,
                (record.memory_id, record.kind.value, record.text, json.dumps(metadata, default=str)),
            )
            if self._fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (record.memory_id,))
                conn.execute(
                    "INSERT INTO memories_fts(memory_id, kind, text, metadata_json) VALUES (?, ?, ?, ?)",
                    (record.memory_id, record.kind.value, record.text, json.dumps(metadata, default=str)),
                )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [
            "".join(ch for ch in term if ch.isalnum() or ch in {"_", "-", ":", ".", "/"})
            for term in query.split()
        ]
        terms = [term for term in terms if term]
        return " OR ".join(f'"{term}"' for term in terms[:32])

    def _candidate_rows(self, query: str, limit: int) -> list[tuple[Any, ...]]:
        expression = self._fts_query(query)
        if self._fts_enabled and expression:
            try:
                with self._connect() as conn:
                    return conn.execute(
                        """
                        SELECT memory_id, kind, text, metadata_json, bm25(memories_fts)
                        FROM memories_fts
                        WHERE memories_fts MATCH ?
                        ORDER BY bm25(memories_fts)
                        LIMIT ?
                        """,
                        (expression, max(limit * 12, 100)),
                    ).fetchall()
            except sqlite3.OperationalError:
                pass

        with self._connect() as conn:
            return [
                (*row, 0.0)
                for row in conn.execute(
                    "SELECT memory_id, kind, text, metadata_json FROM memories LIMIT ?",
                    (max(limit * 20, 200),),
                ).fetchall()
            ]

    @staticmethod
    def _rerank_score(query: str, memory_id: str, kind: str, text: str, metadata: dict[str, Any], fts_rank: float) -> float:
        q = _tokens(query)
        t = _tokens(f"{text} {metadata}")
        lexical = len(q & t) / max(1, len(q))

        q_ids = _identifier_tokens(query)
        record_ids = _identifier_tokens(f"{memory_id} {text} {metadata}")
        exact_id = 1.0 if q_ids and q_ids.intersection(record_ids) else 0.0

        phrase = 1.0 if query.strip().lower() and query.strip().lower() in text.lower() else 0.0

        confidence = metadata.get("confidence", metadata.get("source_confidence", 0.5))
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5

        hits = max(0, int(metadata.get("_memory_hits", 0) or 0))
        age_penalty = 0.0
        updated = str(metadata.get("_memory_updated_at") or "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                age_days = max(0.0, (datetime.now(UTC) - dt.astimezone(UTC)).total_seconds() / 86400.0)
                age_penalty = min(0.18, age_days / 3650.0)
            except ValueError:
                pass
        recency = max(0.0, 1.0 - age_penalty)
        stability = min(1.0, math.log1p(hits) / math.log(11.0))

        # FTS is a candidate generator, not the final ranking authority.
        fts_component = 1.0 / (1.0 + abs(float(fts_rank or 0.0)))
        kind_weight = _KIND_WEIGHT.get(kind, 0.90)

        score = (
            0.45 * lexical
            + 0.20 * fts_component
            + 0.15 * exact_id
            + 0.05 * phrase
            + 0.07 * confidence
            + 0.05 * recency
            + 0.03 * stability
        )
        return score * kind_weight

    def _touch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = _now()
        with self._connect() as conn:
            for memory_id in memory_ids:
                row = conn.execute(
                    "SELECT metadata_json FROM memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if not row:
                    continue
                metadata = json.loads(row[0])
                metadata["_memory_last_accessed"] = now
                metadata["_memory_hits"] = int(metadata.get("_memory_hits", 0) or 0) + 1
                conn.execute(
                    "UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
                    (json.dumps(metadata, default=str), memory_id),
                )
                if self._fts_enabled:
                    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
                    kind_row = conn.execute("SELECT kind, text FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
                    if kind_row:
                        conn.execute(
                            "INSERT INTO memories_fts(memory_id, kind, text, metadata_json) VALUES (?, ?, ?, ?)",
                            (memory_id, kind_row[0], kind_row[1], json.dumps(metadata, default=str)),
                        )

    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []

        rows = self._candidate_rows(query, limit)
        kind_values = {kind.value for kind in kinds} if kinds else None
        scored: list[tuple[float, MemoryRecord]] = []

        for memory_id, kind, text, metadata_json, rank in rows:
            if kind_values and kind not in kind_values:
                continue
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            score = self._rerank_score(query, memory_id, kind, text, metadata, rank)
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    MemoryRecord(
                        memory_id=memory_id,
                        kind=MemoryKind(kind),
                        text=text,
                        metadata=metadata,
                        score=round(score, 4),
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [record for _, record in scored[:limit]]
        self._touch([record.memory_id for record in results])
        return results
