"""Stable public entrypoint for the two-stage Pre-CAB pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import EvidenceDocument, EvidenceResult, verify_attachments
from .final_reasoning import FinalReasoningResult, reconcile_final_decision, run_final_reasoning
from .local_attachments import load_attachments_for_cr
from .orchestrator import run_stage1
from .schemas import Decision, Strictness


@dataclass
class FinalPipelineResult:
    stage1: Any
    stage2: EvidenceResult | None
    final_decision: Decision
    documents: list[EvidenceDocument]
    final_reasoning: FinalReasoningResult | None = None


def _merge_decisions(stage1: Decision, stage2: Decision | None) -> Decision:
    if stage1 == Decision.NOT_READY or stage2 == Decision.NOT_READY:
        return Decision.NOT_READY
    if stage1 == Decision.CONDITIONAL or stage2 == Decision.CONDITIONAL:
        return Decision.CONDITIONAL
    return Decision.PASS


def run_pre_cab(
    cr: dict[str, Any],
    documents: list[EvidenceDocument] | None = None,
    *,
    attachment_root: str | Path | None = None,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
) -> FinalPipelineResult:
    """Evaluate a Normal CR, verify evidence, then run one evidence-aware GPT synthesis.

    Free-tier behavior intentionally avoids a Stage-1 model call. Deterministic specialists first
    decide whether the CR can progress to document verification; GPT-OSS is reserved for the final
    high-value reasoning pass after evidence and historical memory are available.
    """
    stage1_result = run_stage1(cr, strictness=strictness, model=None, memory=memory)
    collected = list(documents or [])

    if attachment_root is not None:
        cr_number = str(cr.get("Number") or cr.get("Effective number") or "").strip()
        collected.extend(load_attachments_for_cr(Path(attachment_root), cr_number))

    deduped: list[EvidenceDocument] = []
    seen: set[str] = set()
    for document in collected:
        if document.ref in seen:
            continue
        seen.add(document.ref)
        deduped.append(document)

    if stage1_result.stage1.decision == Decision.NOT_READY:
        return FinalPipelineResult(
            stage1=stage1_result.stage1,
            stage2=None,
            final_decision=Decision.NOT_READY,
            documents=deduped,
            final_reasoning=None,
        )

    stage2 = verify_attachments(
        cr,
        deduped,
        requirements=stage1_result.stage1.requirements,
        strictness=strictness,
    )
    deterministic_final = _merge_decisions(stage1_result.stage1.decision, stage2.decision)

    reasoning = run_final_reasoning(
        cr,
        stage1_result.stage1,
        stage2,
        strictness=strictness,
        model=model,
        memory=memory,
    )
    final_decision, model_finding = reconcile_final_decision(deterministic_final, reasoning)
    if model_finding is not None:
        stage1_result.stage1.findings.append(model_finding)

    payload = reasoning.payload or {}
    if payload:
        stage1_result.stage1.cab_summary = str(payload.get("cab_reasoning") or stage1_result.stage1.cab_summary)
        stage1_result.stage1.technical_summary = str(payload.get("technical_reasoning") or stage1_result.stage1.technical_summary)
        confidence = payload.get("confidence")
        if isinstance(confidence, (int, float)):
            stage1_result.stage1.confidence = min(
                stage1_result.stage1.confidence,
                max(0.0, min(1.0, float(confidence))),
            )

    stage1_result.stage1.metadata.update(
        {
            "final_reasoning_model": getattr(model, "model_name", None),
            "final_model_prediction": reasoning.model_prediction.value if reasoning.model_prediction else None,
            "final_reasoning_error": reasoning.error,
            "final_reasoning_mode": reasoning.reasoning.mode if reasoning.reasoning else "none",
            "documents_analyzed": len(deduped),
            "deterministic_final": deterministic_final.value,
            "final_decision": final_decision.value,
            "brain": payload,
        }
    )

    return FinalPipelineResult(
        stage1=stage1_result.stage1,
        stage2=stage2,
        final_decision=final_decision,
        documents=deduped,
        final_reasoning=reasoning,
    )
