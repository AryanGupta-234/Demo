"""Runtime GPT-OSS 120B provider bootstrap with .env-compatible multi-provider failover."""
from __future__ import annotations

import os

from .budget import BudgetGuard
from .budgeted_provider import BudgetedProvider
from .cerebras import CerebrasGPTOSS120B
from .huggingface import HuggingFaceGPTOSS120B
from .model_cache import CachingProvider
from .model_router import GPTOSS120BRouter
from .models import GroqGPTOSS120B, ModelProvider


def build_runtime_provider(name: str | None = None) -> ModelProvider:
    selected = (name or os.getenv("LLM_PROVIDER") or os.getenv("PRE_CAB_PROVIDER") or "auto").strip().lower()
    if selected == "cerebras":
        provider: ModelProvider = CerebrasGPTOSS120B()
    elif selected == "groq":
        provider = GroqGPTOSS120B()
    elif selected in {"hf", "huggingface", "hugging-face"}:
        provider = HuggingFaceGPTOSS120B()
    elif selected in {"auto", "failover"}:
        candidates: list[ModelProvider] = []
        if os.getenv("CEREBRAS_API_KEY"):
            candidates.append(CerebrasGPTOSS120B())
        if os.getenv("GROQ_API_KEY"):
            candidates.append(GroqGPTOSS120B())
        if os.getenv("HF_TOKEN"):
            candidates.append(HuggingFaceGPTOSS120B())
        if not candidates:
            raise RuntimeError("No GPT-OSS 120B provider credentials found")
        provider = GPTOSS120BRouter(candidates)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {selected}")

    if os.getenv("PRE_CAB_BUDGET_GUARD", "1") != "0":
        provider = BudgetedProvider(provider, BudgetGuard())
    if os.getenv("PRE_CAB_MODEL_CACHE", "1") != "0":
        provider = CachingProvider(provider, os.getenv("PRE_CAB_MODEL_CACHE_PATH", ".pre_cab/model_cache.sqlite3"))
    return provider
