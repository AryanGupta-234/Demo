"""Stable public entrypoint for the two-stage Pre-CAB pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import EvidenceDocument, EvidenceResult, verify_attachments
from .final_reasoning import FinalReasoningResult, reconcile_final_decision, run_final_reasoning
from .input_loader import normalize_cr_record
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
    agent_results: list[Any] = field(default_factory=list)


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
    """Evaluate a CR through normalization, deterministic agents, optional evidence, and GPT reasoning.

    Normalization happens at the pipeline boundary so callers cannot accidentally bypass canonical
    ServiceNow field aliases. The generative brain still runs for NOT_READY cases so CAB reviewers get
    a complete explanation instead of a checklist-only early exit; deterministic blockers remain final
    safety constraints and the model can only make the outcome more conservative.
    """
    canonical_cr = normalize_cr_record(cr)

    # Specialist agents are deterministic preprocessing; GPT-OSS is reserved for the high-value
    # synthesis pass in ``run_final_reasoning`` to control free-tier usage.
    stage1_result = run_stage1(canonical_cr, strictness=strictness, model=None, memory=memory)
    collected = list(documents or [])

    if attachment_root is not None:
        cr_number = str(
            canonical_cr.get("Number")
            or canonical_cr.get("Effective number")
            or canonical_cr.get("change_number")
            or ""
        ).strip()
        collected.extend(load_attachments_for_cr(Path(attachment_root), cr_number))

    deduped: list[EvidenceDocument] = []
    seen: set[str] = set()
    for document in collected:
        if document.ref in seen:
            continue
        seen.add(document.ref)
        deduped.append(document)

    # Evidence verification is meaningful only when documents are actually supplied. JSON-only
    # validation should not invent an evidence penalty merely because attachments are absent.
    stage2: EvidenceResult | None = None
    if deduped:
        stage2 = verify_attachments(
            canonical_cr,
            deduped,
            requirements=stage1_result.stage1.requirements,
            strictness=strictness,
        )

    deterministic_final = _merge_decisions(
        stage1_result.stage1.decision,
        stage2.decision if stage2 is not None else None,
    )

    # Always run the generative brain, including NOT_READY cases. The model explains the decision,
    # identifies uncertainty and remediation; it cannot upgrade a deterministic blocker.
    reasoning = run_final_reasoning(
        canonical_cr,
        stage1_result.stage1,
        stage2,
        strictness=strictness,
        model=model,
        memory=memory,
        documents=deduped,
    )
    final_decision, model_finding = reconcile_final_decision(deterministic_final, reasoning)
    if model_finding is not None:
        stage1_result.stage1.findings.append(model_finding)

    payload = reasoning.payload or {}
    if payload:
        stage1_result.stage1.cab_summary = str(
            payload.get("cab_reasoning") or stage1_result.stage1.cab_summary
        )
        stage1_result.stage1.technical_summary = str(
            payload.get("technical_reasoning") or stage1_result.stage1.technical_summary
        )
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
        agent_results=stage1_result.agent_results,
    )
