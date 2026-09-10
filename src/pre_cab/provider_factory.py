from __future__ import annotations
import os
from typing import Any
from .models import GroqGPTOSS120B, ModelProvider
from .huggingface import HuggingFaceGPTOSS120B


def build_provider(name: str | None = None) -> ModelProvider:
    """Build the configured GPT-OSS 120B provider without changing agent behavior."""
    provider = (name or os.getenv("PRE_CAB_PROVIDER") or "groq").strip().lower()
    if provider in {"groq", "groq-gpt-oss-120b"}:
        return GroqGPTOSS120B()
    if provider in {"hf", "huggingface", "hugging-face"}:
        return HuggingFaceGPTOSS120B()
    raise ValueError(f"Unsupported provider: {provider}")
