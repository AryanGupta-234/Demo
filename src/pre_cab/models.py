"""Model-provider abstraction using the official Groq SDK for GPT-OSS 120B."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    raw: dict[str, Any]


class ModelProvider(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        ...


class GroqGPTOSS120B:
    """Official Groq SDK adapter for OpenAI GPT-OSS 120B."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required to use the GPT-OSS 120B provider.")

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high for GPT-OSS 120B")
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            kwargs: dict[str, Any] = {
                "model": os.getenv("GROQ_MODEL", self.model_name),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else {}
            return ModelResponse(text=text, model=response.model, raw=raw)
        except Exception as exc:
            raise RuntimeError(f"GPT-OSS 120B request failed: {exc}") from exc
