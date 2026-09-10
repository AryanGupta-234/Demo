"""Groq model adapter. V1 is pinned to GPT-OSS 120B for capability benchmarking."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .models import ModelProvider, ModelResponse


@dataclass
class GroqGPTOSS120B(ModelProvider):
    """OpenAI-compatible Groq adapter with one stable model identity."""

    api_key: str | None = None
    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"

    def generate(self, *, system: str, user: str, temperature: float = 0.1, **kwargs: Any) -> ModelResponse:
        key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use the Groq adapter") from exc

        client = OpenAI(api_key=key, base_url=self.base_url)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        text = completion.choices[0].message.content or ""
        return ModelResponse(text=text, model=self.model, raw=completion.model_dump())
