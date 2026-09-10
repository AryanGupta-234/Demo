from pre_cab.budget import BudgetGuard, InferenceBudget, plan_reasoning_call
from pre_cab.huggingface import HuggingFaceGPTOSS120B
from pre_cab.model_router import GPTOSS120BRouter
from pre_cab.models import ModelResponse


class FakeProvider:
    model_name = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("quota")
        return ModelResponse("ok", self.model_name, {})


def test_huggingface_provider_reference():
    provider = HuggingFaceGPTOSS120B(api_key="test", provider_policy="cheapest")
    assert provider._model_ref() == "openai/gpt-oss-120b:cheapest"


def test_router_falls_back_to_second_provider():
    first = FakeProvider(fail=True)
    second = FakeProvider()
    router = GPTOSS120BRouter([first, second])
    result = router.generate(system="s", user="u")
    assert result.model == "fake"
    assert first.calls == 1
    assert second.calls == 1
    assert router.last_attempts[0].ok is False
    assert router.last_attempts[1].ok is True


def test_budget_guard_rejects_overlarge_call():
    budget = InferenceBudget(max_input_tokens_per_call=10, tokens_per_minute=100)
    plan = plan_reasoning_call("x" * 100, budget=budget)
    assert plan.calls == 0
    guard = BudgetGuard(budget)
    assert guard.allow(estimated_input_tokens=20, estimated_output_tokens=10) is False
