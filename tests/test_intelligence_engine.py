from __future__ import annotations

from pathlib import Path

from pre_cab.agents import CrossAgentConsistencyAgent
from pre_cab.memory import MemoryKind, MemoryRecord, remember_validation_episode
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.schemas import AgentContext, Decision, Finding, FindingSeverity, Strictness


def test_validation_episode_becomes_retrievable(tmp_path: Path) -> None:
    memory = SQLiteUnifiedMemory(tmp_path / "memory.db")
    cr = {
        "Number": "CHG0099001",
        "Type": "Normal",
        "Short description": "Customer API deployment",
    }
    finding = Finding(
        "TEST_GAP",
        "Test evidence missing",
        FindingSeverity.WARNING,
        "Execution result is not recorded.",
    )

    record = remember_validation_episode(
        memory,
        cr,
        decision=Decision.CONDITIONAL.value,
        confidence=0.74,
        findings=[finding],
        profile={"primary_archetype": "FUNCTIONAL", "functional": True},
        reasoning_summary="Test execution remains unverified.",
    )

    results = memory.search("CHG0099001 functional test execution", kinds=[MemoryKind.EPISODE], limit=3)
    assert results
    assert results[0].memory_id == record.memory_id
    assert results[0].metadata["authoritative"] is False


def test_cross_agent_consistency_uses_shared_blackboard() -> None:
    state = {
        "context": {
            "notes": {
                "change_profile": {
                    "functional": True,
                    "customer_facing": True,
                    "high_impact": False,
                }
            }
        },
        "technical": {"notes": {"implementation_present": False}},
        "business_impact": {"notes": {"customer_visible_signal": False}},
        "testing": {"notes": {"evidence_matrix": {"execution_status": "FAILED"}}},
        "risk": {"notes": {"contradiction": False}},
    }
    agent = CrossAgentConsistencyAgent()
    result = agent.run(
        AgentContext(
            cr={"Number": "CHG0099002", "Type": "Normal"},
            strictness=Strictness.BALANCED,
            agent_state=state,
        )
    )

    codes = {finding.code for finding in result.findings}
    assert "AGENT_CONSISTENCY_IMPLEMENTATION" in codes
    assert "AGENT_CONSISTENCY_CUSTOMER_SIGNAL" in codes
    assert "AGENT_CONSISTENCY_TEST_EXECUTION" in codes
