"""Load explicit governance/SOP rules into unified memory as authoritative policy context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .memory import MemoryKind, MemoryRecord, UnifiedMemory


def _records_from_payload(payload: Any, source: str) -> list[MemoryRecord]:
    rows = payload if isinstance(payload, list) else payload.get("rules", []) if isinstance(payload, dict) else []
    records: list[MemoryRecord] = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            rule_id = f"rule-{index + 1}"
            text = row
            metadata: dict[str, Any] = {}
        elif isinstance(row, dict):
            rule_id = str(row.get("id") or row.get("name") or f"rule-{index + 1}")
            text = str(row.get("text") or row.get("rule") or row.get("description") or "").strip()
            metadata = {key: value for key, value in row.items() if key not in {"text", "rule", "description"}}
        else:
            continue
        if not text:
            continue
        records.append(
            MemoryRecord(
                memory_id=f"policy:{source}:{rule_id}",
                kind=MemoryKind.POLICY,
                text=text,
                metadata={"source": source, "authoritative": True, **metadata},
            )
        )
    return records


def load_policy_file(path: str | Path) -> list[MemoryRecord]:
    """Load a JSON policy catalog. Text/PDF policy extraction can feed the same memory contract later."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Policy loader currently accepts JSON catalogs; extracted text can be converted to JSON rules")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _records_from_payload(payload, path.name)


def remember_policies(memory: UnifiedMemory, policies: Iterable[MemoryRecord]) -> int:
    count = 0
    for record in policies:
        if record.kind != MemoryKind.POLICY:
            raise ValueError("Only POLICY records can be loaded through remember_policies")
        memory.remember(record)
        count += 1
    return count
