"""Local Ollama adapter for the Pre-CAB reasoning layer."""
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
        try:
            from urllib.error import URLError
            from urllib.request import Request, urlopen

            # Default to Ollama's practical 32K setting. The model family supports
            # longer context, but the actual Ollama allocation depends on the
            # server/model configuration and available memory. Raise this only
            # after checking `ollama ps` and available VRAM/RAM.
            try:
                num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
            except ValueError:
                num_ctx = 32768
            num_ctx = max(4096, min(num_ctx, 131072))

            payload: dict[str, Any] = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx},
                "format": "json",
            }

            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=300) as response:
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
