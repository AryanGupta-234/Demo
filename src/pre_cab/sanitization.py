"""Minimal prompt sanitization for sensitive runtime values.

This deliberately preserves technical CR content (hosts, components, plans) unless it resembles a
credential/secret. The goal is not to erase useful engineering context; it is to prevent accidental
secret leakage and benchmark-answer leakage.
"""
from __future__ import annotations

import re
from typing import Any

from .benchmark_leakage import strip_post_decision_fields

_SECRET_KEY_PATTERN = re.compile(r"(?i)(api[_ -]?key|secret|password|passwd|token|credential)")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}")
_COMMON_KEY_PATTERN = re.compile(r"\b(?:gsk_|hf_|sk-)[A-Za-z0-9_-]{12,}\b")


def _redact_text(value: str) -> str:
    value = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    value = _COMMON_KEY_PATTERN.sub("[REDACTED_SECRET]", value)
    return value


def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                result[str(key)] = "[REDACTED_SECRET]"
            else:
                result[str(key)] = sanitize_value(item)
        return result
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_value(item) for item in value)
    return value


def sanitize_cr_for_reasoning(cr: dict[str, Any], *, benchmark_mode: bool = False) -> dict[str, Any]:
    data = strip_post_decision_fields(cr) if benchmark_mode else dict(cr)
    return sanitize_value(data)
