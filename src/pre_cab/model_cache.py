"""Local content-addressed cache for GPT-OSS reasoning responses."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelProvider, ModelResponse


def _cache_key(
    model_name: str,
    system: str,
    user: str,
    temperature: float,
    response_format: dict[str, Any] | None,
    reasoning_effort: str,
) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "system": system,
            "user": user,
            "temperature": temperature,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachingProvider:
    provider: ModelProvider
    path: str | Path = ".pre_cab/model_cache.sqlite3"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                )
                """
            )

    @property
    def model_name(self) -> str:
        return getattr(self.provider, "model_name", "unknown")

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "high",
    ) -> ModelResponse:
        key = _cache_key(
            self.model_name,
            system,
            user,
            temperature,
            response_format,
            reasoning_effort,
        )
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT model, text, raw_json FROM model_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row:
            raw = json.loads(row[2])
            raw["pre_cab_cache"] = {"hit": True, "key": key}
            return ModelResponse(text=row[1], model=row[0], raw=raw)

        response = self.provider.generate(
            system=system,
            user=user,
            temperature=temperature,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        raw = dict(response.raw) if isinstance(response.raw, dict) else {}
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_cache(cache_key, model, text, raw_json) VALUES (?, ?, ?, ?)",
                (key, response.model, response.text, json.dumps(raw, default=str)),
            )
        raw["pre_cab_cache"] = {"hit": False, "key": key}
        return ModelResponse(response.text, response.model, raw)
