"""Model-provider abstraction. V1 uses Groq GPT-OSS 120B; agents do not depend on Groq directly."""
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
    model_name: str

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
    """Thin OpenAI-compatible client adapter for Groq's GPT-OSS 120B."""

    model_name = "openai/gpt-oss-120b"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
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
            from urllib.request import Request, urlopen
            import json

            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
            if response_format:
                payload["response_format"] = response_format

            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"].get("content", "")
            return ModelResponse(text=text, model=self.model_name, raw=raw)
        except Exception as exc:
            raise RuntimeError(f"GPT-OSS 120B request failed: {exc}") from exc
