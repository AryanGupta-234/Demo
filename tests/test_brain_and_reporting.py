from dataclasses import dataclass

from pre_cab.brain_loop import AgenticReasoningLoop
from pre_cab.memory import InMemoryUnifiedMemory
from pre_cab.models import ModelResponse
from pre_cab.reporting import build_report
from pre_cab.schemas import AgentContext, Finding, FindingSeverity, Strictness, ValidationResult, Decision


@dataclass
class FakeModel:
    calls: int = 0
    model_name: str = "openai/gpt-oss-120b"

    def generate(self, *, system: str, user: str, temperature: float = 0.1, response_format=None):
        self.calls += 1
        return ModelResponse(
            text=f"pass-{self.calls}",
            model=self.model_name,
            raw={"call": self.calls, "system": system[:40]},
        )


def test_reasoning_loop_runs_initial_and_critique_passes():
    model = FakeModel()
    brain = AgenticReasoningLoop(model, InMemoryUnifiedMemory())
    result = brain.run(AgentContext(cr={"Number": "CHG-DEMO-001"}, strictness=Strictness.BALANCED))
    assert model.calls == 2
    assert result.initial.text == "pass-1"
    assert result.critique.text == "pass-2"
    assert result.questions


def test_report_keeps_cab_and_technical_views():
    finding = Finding(
        code="EXAMPLE",
        title="Example warning",
        severity=FindingSeverity.WARNING,
        message="Something needs attention.",
        technical_detail="Technical detail remains available.",
        recommendation="Review the item.",
    )
    validation = ValidationResult(
        decision=Decision.CONDITIONAL,
        confidence=0.91,
        score=88,
        strictness=Strictness.BALANCED,
        findings=[finding],
        technical_summary="Technical summary.",
        cab_summary="Conditional review.",
    )
    report = build_report(validation)
    assert report["cab_view"]["attention_items"][0]["message"] == finding.message
    assert report["technical_view"]["warnings"][0]["technical_detail"] == finding.technical_detail
