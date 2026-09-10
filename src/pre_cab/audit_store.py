"""Persistent local audit trail for Pre-CAB runs."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRun:
    run_id: str
    cr_number: str
    created_at: str
    strictness: str
    stage1_decision: str
    stage2_decision: str | None
    final_decision: str
    model: str | None
    payload: dict[str, Any]


class SQLiteAuditStore:
    def __init__(self, path: str | Path = ".pre_cab/audit.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_runs (
                    run_id TEXT PRIMARY KEY,
                    cr_number TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    strictness TEXT NOT NULL,
                    stage1_decision TEXT NOT NULL,
                    stage2_decision TEXT,
                    final_decision TEXT NOT NULL,
                    model TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_validation_runs_cr ON validation_runs(cr_number)")

    def record(
        self,
        *,
        cr_number: str,
        strictness: str,
        stage1_decision: str,
        stage2_decision: str | None,
        final_decision: str,
        model: str | None,
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO validation_runs(
                    run_id, cr_number, created_at, strictness, stage1_decision,
                    stage2_decision, final_decision, model, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cr_number,
                    created_at,
                    strictness,
                    stage1_decision,
                    stage2_decision,
                    final_decision,
                    model,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
        return run_id

    def get(self, run_id: str) -> AuditRun | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT run_id, cr_number, created_at, strictness, stage1_decision,
                       stage2_decision, final_decision, model, payload_json
                FROM validation_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return AuditRun(
            run_id=row[0],
            cr_number=row[1],
            created_at=row[2],
            strictness=row[3],
            stage1_decision=row[4],
            stage2_decision=row[5],
            final_decision=row[6],
            model=row[7],
            payload=json.loads(row[8]),
        )

    def recent_for_cr(self, cr_number: str, limit: int = 20) -> list[AuditRun]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT run_id FROM validation_runs
                WHERE cr_number = ? ORDER BY created_at DESC LIMIT ?
                """,
                (cr_number, limit),
            ).fetchall()
        return [run for row in rows if (run := self.get(row[0])) is not None]
