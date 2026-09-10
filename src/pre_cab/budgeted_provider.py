"""Budget-aware wrapper around any GPT-OSS 120B ModelProvider."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .budget import BudgetGuard, estimate_tokens
from .models import ModelProvider, ModelResponse


@dataclass
class BudgetedProvider:
    provider: ModelProvider
    guard: BudgetGuard

    @property
    def model_name(self) -> str:
        return getattr(self.provider, "model_name", "unknown")

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        estimated_input = estimate_tokens(system) + estimate_tokens(user)
        estimated_output = self.guard.budget.reserved_output_tokens
        if not self.guard.allow(
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
        ):
            raise RuntimeError(
                "Free-tier inference budget denied this call. Reduce retrieved context, wait for the "
                "rate window, switch provider, or run deterministic-only mode."
            )

        response = self.provider.generate(
            system=system,
            user=user,
            temperature=temperature,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )

        usage = response.raw.get("usage", {}) if isinstance(response.raw, dict) else {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or estimated_input)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or estimated_output)
        self.guard.record(input_tokens=input_tokens, output_tokens=output_tokens)
        return response
