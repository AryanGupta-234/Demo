"""Dependency-free persistent unified-memory adapter for demo and benchmark runs.

The interface mirrors the existing UnifiedMemory protocol so PostgreSQL/pgvector can replace it later.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .memory import MemoryKind, MemoryRecord


class SQLiteUnifiedMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
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

    def remember(self, record: MemoryRecord) -> None:
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
                (record.memory_id, record.kind.value, record.text, json.dumps(record.metadata, default=str)),
            )

    def search(self, query: str, *, kinds: list[MemoryKind] | None = None, limit: int = 10) -> list[MemoryRecord]:
        terms = [term.lower() for term in query.split() if term.strip()]
        with self._connect() as conn:
            rows = conn.execute("SELECT memory_id, kind, text, metadata_json FROM memories").fetchall()

        kind_values = {kind.value for kind in kinds} if kinds else None
        scored: list[MemoryRecord] = []
        for memory_id, kind, text, metadata_json in rows:
            if kind_values and kind not in kind_values:
                continue
            haystack = f"{text} {metadata_json}".lower()
            score = (sum(term in haystack for term in terms) / len(terms)) if terms else 0.0
            if score > 0:
                scored.append(
                    MemoryRecord(memory_id, MemoryKind(kind), text, json.loads(metadata_json), round(score, 4))
                )
        return sorted(scored, key=lambda item: item.score or 0.0, reverse=True)[:limit]
