"""Provider router for GPT-OSS 120B.

The rest of the application only sees the ModelProvider contract. Groq and Hugging Face can fail over
without changing agents, memory, evidence validation or decision logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import ModelProvider, ModelResponse


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    ok: bool
    error: str = ""


class GPTOSS120BRouter:
    """Failover router for the same GPT-OSS 120B capability across providers."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, providers: Sequence[ModelProvider]) -> None:
        if not providers:
            raise ValueError("At least one GPT-OSS 120B provider is required")
        self.providers = tuple(providers)
        self.last_attempts: tuple[ProviderAttempt, ...] = ()

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        attempts: list[ProviderAttempt] = []
        for provider in self.providers:
            name = f"{provider.__class__.__name__}:{getattr(provider, 'model_name', 'unknown')}"
            try:
                response = provider.generate(
                    system=system,
                    user=user,
                    temperature=temperature,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )
                attempts.append(ProviderAttempt(name, True))
                self.last_attempts = tuple(attempts)
                raw = dict(response.raw) if isinstance(response.raw, dict) else {}
                raw["pre_cab_router"] = {
                    "provider_used": name,
                    "attempts": [attempt.__dict__ for attempt in attempts],
                }
                return ModelResponse(response.text, response.model, raw)
            except Exception as exc:
                attempts.append(ProviderAttempt(name, False, str(exc)))
        self.last_attempts = tuple(attempts)
        detail = "; ".join(f"{attempt.provider}: {attempt.error}" for attempt in attempts)
        raise RuntimeError(f"All GPT-OSS 120B providers failed: {detail}")
