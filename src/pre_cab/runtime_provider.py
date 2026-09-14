"""Runtime GPT-OSS 120B provider bootstrap.

V1 uses Groq as the primary cloud provider. Hugging Face remains an optional
fallback; Cerebras is intentionally disabled for now. Ollama is available as an
explicit, local, zero-cost option (never part of "auto" fallback) - useful for
demonstrating the reasoning capability against real CR data before budgeting
cloud GPU/API time for anything, and independent of the training/ fine-tuning
path (a stock instruct model, not the org's own fine-tuned GPT-OSS 20B).
"""
from __future__ import annotations

import os

from .budget import BudgetGuard
from .budgeted_provider import BudgetedProvider
from .huggingface import HuggingFaceGPTOSS120B
from .model_cache import CachingProvider
from .model_router import GPTOSS120BRouter
from .models import GroqGPTOSS120B, ModelProvider
from .ollama_provider import OllamaProvider


def build_runtime_provider(name: str | None = None) -> ModelProvider:
    selected = (name or os.getenv("LLM_PROVIDER") or os.getenv("PRE_CAB_PROVIDER") or "groq").strip().lower()

    skip_budget_guard = False
    if selected == "groq":
        provider: ModelProvider = GroqGPTOSS120B()
    elif selected in {"hf", "huggingface", "hugging-face"}:
        provider = HuggingFaceGPTOSS120B()
    elif selected == "ollama":
        provider = OllamaProvider()
        skip_budget_guard = True  # local, free - nothing to meter
    elif selected in {"auto", "failover"}:
        candidates: list[ModelProvider] = []
        if os.getenv("GROQ_API_KEY"):
            candidates.append(GroqGPTOSS120B())
        if os.getenv("HF_TOKEN"):
            candidates.append(HuggingFaceGPTOSS120B())
        if not candidates:
            raise RuntimeError("No Groq/Hugging Face GPT-OSS 120B provider credentials found")
        provider = GPTOSS120BRouter(candidates)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER for V1: {selected}. Use groq, huggingface, ollama, or auto"
        )

    if not skip_budget_guard and os.getenv("PRE_CAB_BUDGET_GUARD", "1") != "0":
        provider = BudgetedProvider(provider, BudgetGuard())
    if os.getenv("PRE_CAB_MODEL_CACHE", "1") != "0":
        provider = CachingProvider(
            provider,
            os.getenv("PRE_CAB_MODEL_CACHE_PATH", ".pre_cab/model_cache.sqlite3"),
        )
    return provider
