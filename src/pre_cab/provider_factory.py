from __future__ import annotations

import os

from .budget import BudgetGuard
from .budgeted_provider import BudgetedProvider
from .huggingface import HuggingFaceGPTOSS120B
from .model_cache import CachingProvider
from .model_router import GPTOSS120BRouter
from .models import GroqGPTOSS120B, ModelProvider


def _raw_provider(name: str) -> ModelProvider:
    if name in {"groq", "groq-gpt-oss-120b"}:
        return GroqGPTOSS120B()
    if name in {"hf", "huggingface", "hugging-face"}:
        return HuggingFaceGPTOSS120B()
    if name in {"auto", "failover"}:
        providers: list[ModelProvider] = []
        if os.getenv("GROQ_API_KEY"):
            providers.append(GroqGPTOSS120B())
        if os.getenv("HF_TOKEN"):
            providers.append(HuggingFaceGPTOSS120B())
        if not providers:
            raise RuntimeError("Set GROQ_API_KEY and/or HF_TOKEN for GPT-OSS 120B inference")
        return GPTOSS120BRouter(providers)
    raise ValueError(f"Unsupported provider: {name}")


def build_provider(
    name: str | None = None,
    *,
    budgeted: bool | None = None,
    cached: bool | None = None,
) -> ModelProvider:
    """Build GPT-OSS 120B inference without changing agent behavior.

    ``auto`` prefers Groq and falls back to Hugging Face. Budget admission control and local response
    caching are enabled by default. Cache wraps the budgeted provider so a cache hit consumes no
    provider quota.
    """
    provider_name = (name or os.getenv("PRE_CAB_PROVIDER") or "auto").strip().lower()
    provider: ModelProvider = _raw_provider(provider_name)

    if budgeted is None:
        budgeted = os.getenv("PRE_CAB_BUDGET_GUARD", "1") != "0"
    if budgeted:
        provider = BudgetedProvider(provider, BudgetGuard())

    if cached is None:
        cached = os.getenv("PRE_CAB_MODEL_CACHE", "1") != "0"
    if cached:
        cache_path = os.getenv("PRE_CAB_MODEL_CACHE_PATH", ".pre_cab/model_cache.sqlite3")
        provider = CachingProvider(provider, cache_path)

    return provider
