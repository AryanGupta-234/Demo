"""Demo bootstrap utilities: seed unified memory with synthetic history only."""
from __future__ import annotations

from .memory import InMemoryUnifiedMemory, MemoryKind, MemoryRecord


def demo_memory() -> InMemoryUnifiedMemory:
    memory = InMemoryUnifiedMemory()
    samples = [
        MemoryRecord("hist-001", MemoryKind.CAB_HISTORY, "Normal debit card transaction enhancement approved after UAT and customer approval.", {"category": "Debit Card", "outcome": "Approved"}),
        MemoryRecord("hist-002", MemoryKind.CAB_HISTORY, "Normal application deployment conditionally approved; customer approval evidence was requested.", {"category": "Core", "outcome": "Conditional"}),
        MemoryRecord("policy-001", MemoryKind.POLICY, "Infrastructure OS patching does not automatically require UAT; appropriate server sanity validation may be sufficient.", {"topic": "testing"}),
        MemoryRecord("policy-002", MemoryKind.POLICY, "Customer-facing functional changes generally require appropriate functional validation and approval evidence.", {"topic": "testing"}),
        MemoryRecord("sim-001", MemoryKind.SIMILARITY, "Normal SMS application enhancement using an application artifact and service restart; tested in UAT.", {"similarity_hint": "high", "cab": "Approved"}),
    ]
    for record in samples:
        memory.remember(record)
    return memory
