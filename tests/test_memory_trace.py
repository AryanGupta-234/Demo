from pathlib import Path

from pre_cab.memory import MemoryKind, MemoryRecord
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.trace import EvaluationTrace


def test_sqlite_unified_memory_round_trip(tmp_path: Path):
    memory = SQLiteUnifiedMemory(tmp_path / "memory.db")
    memory.remember(MemoryRecord("cab-1", MemoryKind.CAB_HISTORY, "Normal customer approval was verified.", {"outcome": "Approved"}))
    matches = memory.search("customer approval", kinds=[MemoryKind.CAB_HISTORY])
    assert matches
    assert matches[0].memory_id == "cab-1"
    assert matches[0].metadata["outcome"] == "Approved"


def test_trace_can_be_serialized():
    trace = EvaluationTrace("CHG-DEMO-001", "balanced")
    trace.add("field", "validate", output_summary={"decision": "PASS"})
    trace.add("brain", "critique", output_summary={"contradictions": []})
    payload = trace.to_dict()
    assert payload["cr_number"] == "CHG-DEMO-001"
    assert len(payload["steps"]) == 2
