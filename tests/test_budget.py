from pre_cab.budget import InferenceBudget, budget_safe, estimate_tokens, plan_reasoning_call


def test_small_payload_gets_one_reasoning_call():
    plan = plan_reasoning_call("short payload")
    assert plan.calls == 1
    assert plan.cacheable is True
    assert plan.mode == "single_reasoning_pass"


def test_large_payload_is_retrieved_down_before_llm():
    budget = InferenceBudget(max_input_tokens_per_call=100)
    plan = plan_reasoning_call("x" * 1000, budget=budget)
    assert plan.calls == 0
    assert plan.mode == "retrieve_more_then_call"


def test_budget_safe_respects_daily_and_minute_token_limits():
    assert budget_safe(1, 3000, 1000) is True
    assert budget_safe(1, 7900, 200) is False
    assert budget_safe(1001, 100, 100) is False


def test_estimate_tokens_is_nonzero():
    assert estimate_tokens("") >= 1
