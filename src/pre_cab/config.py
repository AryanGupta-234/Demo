"""Runtime configuration with safe defaults for demo, benchmark and provider routing."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .schemas import Strictness


@dataclass(frozen=True)
class Settings:
    strictness: Strictness = Strictness.BALANCED
    model_name: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_provider: str = "fastest"
    provider_order: tuple[str, ...] = ("groq", "huggingface")
    max_memory_results: int = 8
    allow_emergency: bool = False
    free_mode: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        raw_strictness = os.getenv("PRE_CAB_STRICTNESS", Strictness.BALANCED.value).lower()
        try:
            strictness = Strictness(raw_strictness)
        except ValueError as exc:
            raise ValueError(f"Invalid PRE_CAB_STRICTNESS={raw_strictness!r}") from exc
        raw_order = os.getenv("PRE_CAB_PROVIDER_ORDER", "groq,huggingface")
        provider_order = tuple(x.strip().lower() for x in raw_order.split(",") if x.strip())
        if not provider_order:
            raise ValueError("PRE_CAB_PROVIDER_ORDER must contain at least one provider")
        return cls(
            strictness=strictness,
            model_name=os.getenv("PRE_CAB_MODEL", "openai/gpt-oss-120b"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            hf_base_url=os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"),
            hf_provider=os.getenv("HF_PROVIDER", "fastest"),
            provider_order=provider_order,
            max_memory_results=int(os.getenv("PRE_CAB_MAX_MEMORY_RESULTS", "8")),
            allow_emergency=False,
            free_mode=os.getenv("PRE_CAB_FREE_MODE", "true").lower() in {"1", "true", "yes", "on"},
        )
