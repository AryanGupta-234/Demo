"""Optional FastAPI surface for the Pre-CAB engine.

This adapter keeps the core package usable without FastAPI installed. It accepts a single CR and
returns the same stable report used by a future UI or ServiceNow integration.
"""
from __future__ import annotations

from typing import Any

from .pipeline import run_pre_cab
from .reporting import build_report
from .schemas import Strictness


def create_app(*, model: Any = None, memory: Any = None) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install the 'api' extra to run the HTTP interface") from exc

    app = FastAPI(title="Pre-CAB Validator", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_configured": model is not None}

    @app.post("/v1/pre-cab/validate")
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        cr = payload.get("cr")
        if not isinstance(cr, dict):
            return {"error": "payload.cr must be an object"}
        raw_strictness = str(payload.get("strictness", Strictness.BALANCED.value)).lower()
        try:
            strictness = Strictness(raw_strictness)
        except ValueError:
            return {"error": f"invalid strictness: {raw_strictness}"}
        result = run_pre_cab(cr, strictness=strictness, model=model, memory=memory)
        return build_report(result.stage1, stage2=result.stage2)

    return app
