"""Hugging Face Inference Providers adapter for GPT-OSS 120B.

Hugging Face exposes an OpenAI-compatible router and can route GPT-OSS 120B to a
specific provider or a provider-selection policy. The adapter implements the same
ModelProvider contract as the Groq adapter so agents remain provider-neutral.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .models import ModelResponse


@dataclass
class HuggingFaceGPTOSS120B:
    """GPT-OSS 120B through Hugging Face Inference Providers."""

    api_key: str | None = None
    model_id: str | None = None
    base_url: str | None = None
    provider_policy: str | None = None

    model_name = "openai/gpt-oss-120b"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("HF_TOKEN")
        self.model_id = self.model_id or os.getenv("HF_MODEL", self.model_name)
        self.base_url = (self.base_url or os.getenv("HF_BASE_URL") or "https://router.huggingface.co/v1").rstrip("/")
        self.provider_policy = self.provider_policy or os.getenv("HF_PROVIDER", "fastest")
        if not self.api_key:
            raise RuntimeError("HF_TOKEN is required to use the Hugging Face GPT-OSS 120B provider.")

    def _model_ref(self) -> str:
        policy = (self.provider_policy or "").strip()
        if not policy:
            return self.model_id or self.model_name
        return f"{self.model_id}:{policy}"

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        try:
            from urllib.request import Request, urlopen

            payload: dict[str, Any] = {
                "model": self._model_ref(),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
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
            return ModelResponse(text=text, model=self._model_ref(), raw=raw)
        except Exception as exc:
            raise RuntimeError(f"Hugging Face GPT-OSS 120B request failed: {exc}") from exc
