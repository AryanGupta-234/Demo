"""Provider failover for GPT-OSS 120B free-tier operation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .models import ModelProvider, ModelResponse


@dataclass
class FailoverProvider:
    providers: list[ModelProvider]

    @property
    def model_name(self) -> str:
        return "failover:" + ",".join(getattr(provider, "model_name", "unknown") for provider in self.providers)

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                response = provider.generate(
                    system=system,
                    user=user,
                    temperature=temperature,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )
                raw = dict(response.raw)
                raw.setdefault("pre_cab_failover", {})
                raw["pre_cab_failover"]["attempt_errors"] = list(errors)
                raw["pre_cab_failover"]["provider_used"] = type(provider).__name__
                return ModelResponse(response.text, response.model, raw)
            except Exception as exc:
                errors.append(f"{type(provider).__name__}: {exc}")
        raise RuntimeError("All GPT-OSS 120B providers failed: " + " | ".join(errors))
