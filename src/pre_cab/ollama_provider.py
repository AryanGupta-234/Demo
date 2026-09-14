"""Local Ollama adapter for the reasoning layer.

Purpose: demonstrate the Pre-CAB reasoning capability against real CR data at
zero cost, before asking anyone to budget cloud GPU time for fine-tuning. Ollama
runs entirely on the local machine (http://localhost:11434 by default) with
whatever model is already pulled - e.g. `ollama list` showing qwen2.5:7b-instruct.

This does NOT require GROQ_API_KEY, CEREBRAS_API_KEY, or any cloud credential.
It also is not a substitute for the fine-tuned GPT-OSS 20B path in training/ -
a general-purpose 7B instruct model has no exposure to this org's real
historical patterns, so treat its output as a reasoning-quality demo, not a
production-calibrated decision source, until/unless it's evaluated the same way
(training/evaluate.py's methodology) against real holdout outcomes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .models import ModelResponse


@dataclass
class OllamaProvider:
    """Adapter for a locally-running Ollama server's /api/chat endpoint."""

    model_id: str | None = None
    base_url: str | None = None

    model_name = "qwen2.5:7b-instruct"

    def __post_init__(self) -> None:
        model_id: str = str(self.model_id or os.getenv("OLLAMA_MODEL", self.model_name))
        base_url: str = str(self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.model_name = model_id

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        # Ollama has no reasoning_effort concept (that's a Groq/Cerebras GPT-OSS
        # dial); accepted here only for ModelProvider protocol compatibility and
        # otherwise ignored.
        try:
            from urllib.error import URLError
            from urllib.request import Request, urlopen

            payload: dict[str, Any] = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            }
            # Ollama's native JSON mode: forces the model to emit a single JSON
            # object rather than free-form text, matching what this pipeline's
            # extract_json()-style parsing everywhere else expects.
            payload["format"] = "json"

            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=180) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                raise RuntimeError(
                    f"Could not reach Ollama at {self.base_url} - is `ollama serve` running "
                    f"and is '{self.model_id}' pulled (`ollama list`)? {exc}"
                ) from exc
            text = raw.get("message", {}).get("content", "")
            return ModelResponse(text=text, model=self.model_name, raw=raw)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
