"""Memory consolidation and feedback helpers.

This layer intentionally stays deterministic: consolidation organizes experience;
it never turns historical outcomes into current policy.
"""
from __future__ import annotations

from typing import Any

from .memory import MemoryRecord


def consolidate_memory(memory: Any, *, max_records: int = 96, threshold: float = 0.88) -> int:
    method = getattr(memory, "consolidate", None)
    if not callable(method):
        return 0
    try:
        return int(method(max_records=max_records, threshold=threshold))
    except Exception:
        return 0


def reinforce_memory(memory: Any, record: MemoryRecord, *, correct: bool, note: str = "") -> None:
    method = getattr(memory, "apply_feedback", None)
    if not callable(method):
        return
    try:
        method(record.memory_id, reward=1.0 if correct else -1.0, note=note)
    except Exception:
        return
