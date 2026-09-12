"""Cerebras Cloud adapter for GPT-OSS 120B."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .models import ModelResponse


@dataclass
class CerebrasGPTOSS120B:
    """OpenAI-compatible Cerebras adapter."""

    api_key: str | None = None
    model_id: str | None = None
    base_url: str | None = None

    model_name = "gpt-oss-120b"

    def __post_init__(self) -> None:
        api_key = self.api_key or os.getenv("CEREBRAS_API_KEY")
        model_id: str = str(self.model_id or os.getenv("CEREBRAS_MODEL", self.model_name))
        base_url: str = str(self.base_url or os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"))
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.model_name = model_id
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY is required for the Cerebras GPT-OSS 120B provider.")

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
            raise ValueError("reasoning_effort must be low, medium, or high")
        try:
            from urllib.request import Request, urlopen

            payload: dict[str, Any] = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            if response_format:
                payload["response_format"] = response_format
            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"].get("content", "")
            return ModelResponse(text=text, model=self.model_name, raw=raw)
        except Exception as exc:
            raise RuntimeError(f"Cerebras GPT-OSS 120B request failed: {exc}") from exc
