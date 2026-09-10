from __future__ import annotations
import os
from .models import GroqGPTOSS120B, ModelProvider
from .huggingface import HuggingFaceGPTOSS120B
from .failover_provider import FailoverProvider


def build_provider(name: str | None = None) -> ModelProvider:
    """Build GPT-OSS 120B inference without changing agent behavior.

    ``auto`` prefers Groq and falls back to Hugging Face when both credentials are present.
    Explicit ``groq`` or ``huggingface`` keeps provider selection deterministic for benchmarking.
    """
    provider = (name or os.getenv("PRE_CAB_PROVIDER") or "auto").strip().lower()

    if provider in {"groq", "groq-gpt-oss-120b"}:
        return GroqGPTOSS120B()
    if provider in {"hf", "huggingface", "hugging-face"}:
        return HuggingFaceGPTOSS120B()
    if provider in {"auto", "failover"}:
        providers: list[ModelProvider] = []
        if os.getenv("GROQ_API_KEY"):
            providers.append(GroqGPTOSS120B())
        if os.getenv("HF_TOKEN"):
            providers.append(HuggingFaceGPTOSS120B())
        if not providers:
            raise RuntimeError("Set GROQ_API_KEY and/or HF_TOKEN for GPT-OSS 120B inference")
        return FailoverProvider(providers)
    raise ValueError(f"Unsupported provider: {provider}")
