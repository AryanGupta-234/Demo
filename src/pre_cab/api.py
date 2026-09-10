"""Optional FastAPI surface for the Pre-CAB engine.

The API accepts CR fields and extracted attachment records. The response exposes the true final
readiness after deterministic validation, evidence verification and optional GPT-OSS reasoning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence import EvidenceDocument
from .pipeline import run_pre_cab
from .reporting import build_report
from .schemas import Strictness


def create_app(*, model: Any = None, memory: Any = None) -> Any:
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install the 'api' extra to run the HTTP interface") from exc

    app = FastAPI(title="Pre-CAB Validator", version="0.5.0")

    @app.get("/")
    def home() -> Any:
        ui = Path(__file__).resolve().parents[2] / "web" / "index.html"
        return FileResponse(ui) if ui.exists() else {"service": "pre-cab-validator"}

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

        raw_documents = payload.get("documents", [])
        if not isinstance(raw_documents, list):
            return {"error": "payload.documents must be an array"}

        documents: list[EvidenceDocument] = []
        for index, raw in enumerate(raw_documents):
            if not isinstance(raw, dict):
                return {"error": f"documents[{index}] must be an object"}
            if not raw.get("text"):
                return {"error": f"documents[{index}].text is required"}
            documents.append(
                EvidenceDocument(
                    ref=str(raw.get("ref") or raw.get("name") or f"document-{index}"),
                    name=str(raw.get("name") or raw.get("ref") or f"document-{index}"),
                    text=str(raw["text"]),
                    document_type=str(raw.get("document_type") or "unknown"),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )

        result = run_pre_cab(
            cr,
            documents=documents,
            strictness=strictness,
            model=model,
            memory=memory,
        )
        report = build_report(result.stage1, stage2=result.stage2)
        report["stage1_decision"] = result.stage1.decision.value
        report["final_decision"] = result.final_decision.value
        report["decision"] = result.final_decision.value
        report["stage2_executed"] = result.stage2 is not None
        report["documents_analyzed"] = len(result.documents)
        report["reasoning"] = {
            "executed": bool(result.final_reasoning and result.final_reasoning.reasoning),
            "prediction": (
                result.final_reasoning.model_prediction.value
                if result.final_reasoning and result.final_reasoning.model_prediction
                else None
            ),
            "error": result.final_reasoning.error if result.final_reasoning else None,
            "mode": (
                result.final_reasoning.reasoning.mode
                if result.final_reasoning and result.final_reasoning.reasoning
                else "none"
            ),
        }
        return report

    return app
