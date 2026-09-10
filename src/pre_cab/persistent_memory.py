"""Dependency-free persistent unified-memory adapter for demo and benchmark runs.

Uses SQLite FTS5 when available so large historical memory does not require loading every record into
Python for each query. Falls back to deterministic lexical scoring on SQLite builds without FTS5.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .memory import MemoryKind, MemoryRecord


class SQLiteUnifiedMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

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
                # Backfill once for databases created before FTS support was added.
                existing = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                if existing == 0:
                    conn.execute(
                        "INSERT INTO memories_fts(memory_id, kind, text, metadata_json) "
                        "SELECT memory_id, kind, text, metadata_json FROM memories"
                    )
            except sqlite3.OperationalError:
                self._fts_enabled = False

    def remember(self, record: MemoryRecord) -> None:
        metadata_json = json.dumps(record.metadata, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(memory_id, kind, text, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind=excluded.kind,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json
                """,
                (record.memory_id, record.kind.value, record.text, metadata_json),
            )
            if self._fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (record.memory_id,))
                conn.execute(
                    "INSERT INTO memories_fts(memory_id, kind, text, metadata_json) VALUES (?, ?, ?, ?)",
                    (record.memory_id, record.kind.value, record.text, metadata_json),
                )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = ["".join(ch for ch in term if ch.isalnum() or ch in {"_", "-"}) for term in query.split()]
        terms = [term for term in terms if term]
        return " OR ".join(f'"{term}"' for term in terms[:20])

    def _fts_search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None,
        limit: int,
    ) -> list[MemoryRecord]:
        expression = self._fts_query(query)
        if not expression:
            return []
        kind_values = {kind.value for kind in kinds} if kinds else None
        # Fetch extra rows before kind filtering to avoid starving results when multiple memory kinds coexist.
        fetch_limit = max(limit * 5, 50)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, kind, text, metadata_json, bm25(memories_fts)
                FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                (expression, fetch_limit),
            ).fetchall()
        results: list[MemoryRecord] = []
        for memory_id, kind, text, metadata_json, rank in rows:
            if kind_values and kind not in kind_values:
                continue
            # bm25 is lower-is-better and commonly negative in SQLite FTS; map monotonically to 0..1.
            score = 1.0 / (1.0 + abs(float(rank or 0.0)))
            results.append(
                MemoryRecord(memory_id, MemoryKind(kind), text, json.loads(metadata_json), round(score, 4))
            )
            if len(results) >= limit:
                break
        return results

    def _lexical_search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None,
        limit: int,
    ) -> list[MemoryRecord]:
        terms = [term.lower() for term in query.split() if term.strip()]
        with self._connect() as conn:
            if kinds:
                values = [kind.value for kind in kinds]
                placeholders = ",".join("?" for _ in values)
                rows = conn.execute(
                    f"SELECT memory_id, kind, text, metadata_json FROM memories WHERE kind IN ({placeholders})",
                    values,
                ).fetchall()
            else:
                rows = conn.execute("SELECT memory_id, kind, text, metadata_json FROM memories").fetchall()

        scored: list[MemoryRecord] = []
        for memory_id, kind, text, metadata_json in rows:
            haystack = f"{text} {metadata_json}".lower()
            score = (sum(term in haystack for term in terms) / len(terms)) if terms else 0.0
            if score > 0:
                scored.append(
                    MemoryRecord(memory_id, MemoryKind(kind), text, json.loads(metadata_json), round(score, 4))
                )
        return sorted(scored, key=lambda item: item.score or 0.0, reverse=True)[:limit]

    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        if self._fts_enabled:
            try:
                return self._fts_search(query, kinds=kinds, limit=limit)
            except sqlite3.OperationalError:
                pass
        return self._lexical_search(query, kinds=kinds, limit=limit)
