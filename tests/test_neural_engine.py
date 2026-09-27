from pre_cab.agents import AgentResult
from pre_cab.agent_intelligence import build_agent_intelligence
from pre_cab.brain_loop import AgenticReasoningLoop
from pre_cab.schemas import Finding, FindingSeverity


def _finding(code: str, severity: FindingSeverity) -> Finding:
    return Finding(
        code=code,
        title=code,
        severity=severity,
        message=f"message for {code}",
    )


def test_agent_intelligence_surfaces_cross_agent_agreement():
    results = [
        AgentResult("technical", [_finding("ROLLBACK_GAP", FindingSeverity.WARNING)], [], {"chain": ["rollback is unclear"]}),
        AgentResult("testing", [_finding("ROLLBACK_GAP", FindingSeverity.WARNING)], [], {"chain": ["rollback evidence remains uncertain"]}),
    ]
    intelligence = build_agent_intelligence(results)

    repeated = intelligence["consensus"]["repeated_finding_codes"]
    assert any(item["code"] == "ROLLBACK_GAP" and item["agreement_count"] == 2 for item in repeated)


def test_narrative_parser_accepts_narrative_only_contract():
    payload = {
        "technical_reasoning": "Technical reasoning.",
        "cab_reasoning": "PASS because the evidence is consistent.",
        "cab_questions": ["What remains to be verified?"],
        "recommendations": ["Attach the referenced evidence."],
        "self_critique": ["No unsupported claim was introduced."],
    }
    parsed = AgenticReasoningLoop._parse_narrative(__import__("json").dumps(payload))
    assert parsed == payload


def test_narrative_parser_rejects_reasoning_decision_fields():
    payload = {
        "prediction": "PASS",
        "confidence": 0.9,
        "technical_reasoning": "Technical reasoning.",
        "cab_reasoning": "PASS.",
        "cab_questions": [],
        "recommendations": [],
        "self_critique": [],
    }
    parsed = AgenticReasoningLoop._parse_narrative(__import__("json").dumps(payload))
    assert parsed is None
