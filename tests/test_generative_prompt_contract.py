from pre_cab.brain import build_narrative_system_prompt, build_reasoning_system_prompt
from pre_cab.brain_loop import AgenticReasoningLoop
from pre_cab.memory import InMemoryUnifiedMemory
from pre_cab.models import ModelResponse
from pre_cab.schemas import AgentContext, Strictness


def test_reasoning_prompt_has_explicit_reviewer_sequence():
    prompt = build_reasoning_system_prompt().lower()
    for term in (
        "phase 1 — understand the change",
        "phase 2 — determine what is applicable",
        "phase 3 — build the current evidence picture",
        "phase 4 — test the testing story",
        "phase 6 — reconcile contradictions",
        "phase 7 — use history correctly",
        "phase 9 — explain the result",
        "claim -> source -> corroboration -> implication -> uncertainty",
        "return only one valid json object",
    ):
        assert term in prompt


def test_reasoning_payload_exposes_authority_and_sequence():
    captured = {}

    class CapturingModel:
        model_name = "prompt-test"

        def generate(self, *, system, user, temperature=0.1, response_format=None, reasoning_effort="high"):
            import json

            captured["system"] = system
            captured["user"] = json.loads(user)
            payload = {
                "prediction": "PASS",
                "confidence": 0.9,
                "facts": ["Implementation plan is present."],
                "inferences": ["The synthetic change has a documented implementation path."],
                "uncertainties": [],
                "contradictions": [],
                "technical_reasoning": "The supplied implementation plan is specific.",
                "cab_reasoning": "The current record contains a documented implementation path.",
                "cab_questions": [],
                "recommendations": [],
                "self_critique": ["Checked for unsupported claims."],
            }
            return ModelResponse(text=json.dumps(payload), model=self.model_name, raw={})

    brain = AgenticReasoningLoop(CapturingModel(), InMemoryUnifiedMemory(), mode="single")
    brain.run(
        AgentContext(
            cr={"Number": "CHG-PROMPT-001", "Implementation plan": "Deploy and smoke test."},
            strictness=Strictness.BALANCED,
        )
    )
    assert captured["user"]["reasoning_sequence"][0] == "understand_change"
    assert captured["user"]["reasoning_sequence"][-1] == "write_audience_specific_explanations"
    assert captured["user"]["source_authority_order"][0] == "deterministic_validation_and_explicit_policy"
    assert captured["user"]["source_authority_order"][-1] == "model_inferences"
    assert "static inspection" in captured["user"]["instruction"]


def test_narrative_prompt_is_presentation_only():
    prompt = build_narrative_system_prompt().lower()
    assert "not allowed to re-decide the cr" in prompt
    assert "never invent" in prompt
    assert "prediction or confidence" in prompt
