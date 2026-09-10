"""Free-tier resource planning for GPT-OSS 120B inference.

V1 keeps specialist work local/deterministic and budgets the LLM for the highest-value reasoning pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class InferenceBudget:
    requests_per_minute: int = 30
    requests_per_day: int = 1000
    tokens_per_minute: int = 8000
    tokens_per_day: int = 200_000
    reserved_output_tokens: int = 1600
    max_input_tokens_per_call: int = 6000


@dataclass(frozen=True)
class InferencePlan:
    calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    cacheable: bool
    mode: str
    reason: str


@dataclass
class UsageState:
    minute_started: float
    minute_requests: int = 0
    minute_tokens: int = 0
    day_started: float = 0.0
    day_requests: int = 0
    day_tokens: int = 0


def estimate_tokens(text: str) -> int:
    """Conservative rough token estimate for budgeting only, not billing."""
    return max(1, (len(text) + 3) // 4)


def plan_reasoning_call(payload_text: str, *, budget: InferenceBudget | None = None) -> InferencePlan:
    budget = budget or InferenceBudget()
    estimated_input = estimate_tokens(payload_text)
    total = estimated_input + budget.reserved_output_tokens
    if estimated_input <= budget.max_input_tokens_per_call and total <= budget.tokens_per_minute:
        return InferencePlan(
            calls=1,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=budget.reserved_output_tokens,
            cacheable=True,
            mode="single_reasoning_pass",
            reason="Keep specialist analysis local and spend one model call on synthesis/self-critique.",
        )

    return InferencePlan(
        calls=0,
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=0,
        cacheable=False,
        mode="retrieve_more_then_call",
        reason="Context is too large for the configured free-tier call budget; retrieve/rerank fewer records first.",
    )


def budget_safe(requests: int, input_tokens: int, output_tokens: int, *, budget: InferenceBudget | None = None) -> bool:
    budget = budget or InferenceBudget()
    return (
        requests <= budget.requests_per_day
        and input_tokens + output_tokens <= budget.tokens_per_day
        and input_tokens + output_tokens <= budget.tokens_per_minute
    )


class BudgetGuard:
    """In-process admission control for development/free-tier model calls."""

    def __init__(self, budget: InferenceBudget | None = None) -> None:
        self.budget = budget or InferenceBudget()
        now = monotonic()
        self.state = UsageState(minute_started=now, day_started=now)

    def _roll_windows(self) -> None:
        now = monotonic()
        if now - self.state.minute_started >= 60:
            self.state.minute_started = now
            self.state.minute_requests = 0
            self.state.minute_tokens = 0
        if now - self.state.day_started >= 86_400:
            self.state.day_started = now
            self.state.day_requests = 0
            self.state.day_tokens = 0

    def allow(self, *, estimated_input_tokens: int, estimated_output_tokens: int) -> bool:
        self._roll_windows()
        total = estimated_input_tokens + estimated_output_tokens
        return (
            self.state.minute_requests < self.budget.requests_per_minute
            and self.state.minute_tokens + total <= self.budget.tokens_per_minute
            and self.state.day_requests < self.budget.requests_per_day
            and self.state.day_tokens + total <= self.budget.tokens_per_day
        )

    def record(self, *, input_tokens: int, output_tokens: int) -> None:
        self._roll_windows()
        total = input_tokens + output_tokens
        self.state.minute_requests += 1
        self.state.minute_tokens += total
        self.state.day_requests += 1
        self.state.day_tokens += total
