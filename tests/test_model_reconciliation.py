from __future__ import annotations

import json

from pre_cab.model_router import GPTOSS120BRouter
from pre_cab.models import ModelResponse
from pre_cab.orchestrator import run_stage1
from pre_cab.schemas import Decision, Strictness


class StaticProvider:
    model_name = "test-gpt-oss-120b"

    def __init__(self, prediction: str = "PASS", fail: bool = False) -> None:
        self.prediction = prediction
        self.fail = fail

    def generate(self, **kwargs):
        if self.fail:
            raise RuntimeError("simulated quota failure")
        payload = {
            "prediction": self.prediction,
            "confidence": 0.9,
            "facts": [],
            "inferences": [],
            "uncertainties": [],
            "contradictions": [],
            "technical_reasoning": "technical reasoning",
            "cab_reasoning": "cab reasoning",
            "cab_questions": [],
            "recommendations": ["review evidence"],
            "self_critique": [],
        }
        return ModelResponse(json.dumps(payload), self.model_name, {})


def _good_cr() -> dict:
    return {
        "Number": "CHG-DEMO-200",
        "Type": "Normal",
        "Short description": "Infrastructure configuration refresh",
        "Description": "Routine configuration refresh with no customer behavior change",
        "Justification": "Operational maintenance",
        "Implementation plan": "Back up configuration and apply change",
        "Backout plan": "Restore backup",
        "Test plan": "Perform service health checks",
        "Conflict status": "No Conflict",
        "Configuration item": "Demo CI",
    }


def test_model_can_downgrade_deterministic_pass() -> None:
    result = run_stage1(_good_cr(), strictness=Strictness.BALANCED, model=StaticProvider("NOT_READY"))
    assert result.stage1.decision == Decision.NOT_READY
    assert result.stage1.metadata["model_prediction"] == "NOT_READY"
    assert any(f.code == "MODEL_CONSERVATIVE_DOWNGRADE" for f in result.stage1.findings)


def test_model_cannot_upgrade_deterministic_blocker() -> None:
    cr = _good_cr()
    cr["Implementation plan"] = ""  # genuinely BLOCKING under BALANCED (baseline field)
    result = run_stage1(cr, strictness=Strictness.BALANCED, model=StaticProvider("PASS"))
    assert result.stage1.decision == Decision.NOT_READY
    assert result.stage1.metadata["deterministic_prediction"] == "NOT_READY"


def test_router_uses_second_provider() -> None:
    router = GPTOSS120BRouter([StaticProvider(fail=True), StaticProvider("PASS")])
    response = router.generate(system="x", user="y")
    assert json.loads(response.text)["prediction"] == "PASS"
    assert len(router.last_attempts) == 2
    assert router.last_attempts[0].ok is False
    assert router.last_attempts[1].ok is True
    assert response.raw["pre_cab_router"]["attempts"][0]["ok"] is False
