"""Runtime configuration with safe defaults for the demo and benchmark environments."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .schemas import Strictness


@dataclass(frozen=True)
class Settings:
    strictness: Strictness = Strictness.BALANCED
    model_name: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    max_memory_results: int = 8
    allow_emergency: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        raw_strictness = os.getenv("PRE_CAB_STRICTNESS", Strictness.BALANCED.value).lower()
        try:
            strictness = Strictness(raw_strictness)
        except ValueError as exc:
            raise ValueError(f"Invalid PRE_CAB_STRICTNESS={raw_strictness!r}") from exc
        return cls(
            strictness=strictness,
            model_name=os.getenv("PRE_CAB_MODEL", "openai/gpt-oss-120b"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            max_memory_results=int(os.getenv("PRE_CAB_MAX_MEMORY_RESULTS", "8")),
            allow_emergency=False,
        )
