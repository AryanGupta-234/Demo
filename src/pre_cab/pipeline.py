"""Stable public entrypoint for the multi-level Pre-CAB pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .evidence import EvidenceDocument, EvidenceResult, verify_attachments
from .evidence_workspace import load_and_classify, safe_cr_dir, stage_files
from .final_reasoning import FinalReasoningResult, reconcile_final_decision, run_final_reasoning
from .input_loader import normalize_cr_record
from .orchestrator import run_stage1
from .schemas import Decision, Finding, FindingSeverity, Strictness


@dataclass
class FinalPipelineResult:
    stage1: Any
    stage2: EvidenceResult | None
    final_decision: Decision
    documents: list[EvidenceDocument]
    final_reasoning: FinalReasoningResult | None = None
    agent_results: list[Any] = field(default_factory=list)
    evidence_manifest: dict[str, Any] = field(default_factory=dict)


def _merge_decisions(stage1: Decision, stage2: Decision | None) -> Decision:
    if stage1 == Decision.NOT_READY or stage2 == Decision.NOT_READY:
        return Decision.NOT_READY
    if stage1 == Decision.CONDITIONAL or stage2 == Decision.CONDITIONAL:
        return Decision.CONDITIONAL
    return Decision.PASS


def _evidence_manifest(
    cr_number: str,
    documents: list[EvidenceDocument],
    classifications: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cr_number": cr_number,
        "folder_created": True,
        "document_count": len(documents),
        "documents": [
            {
                "name": d.name,
                "ref": d.ref,
                "type": d.document_type,
                "bytes": d.metadata.get("bytes"),
                "classification": d.metadata.get("evidence_classification")
                or next(
                    (c for c in classifications if c.get("ref") == d.ref),
                    {"type": "other"},
                ),
                "extraction_error": d.metadata.get("extraction_error"),
                "requires_vision": bool(d.metadata.get("requires_vision")),
                "text_chars": len(d.text),
            }
            for d in documents
        ],
    }


def run_pre_cab(
    cr: dict[str, Any],
    documents: list[EvidenceDocument] | None = None,
    *,
    attachment_root: str | Path | None = None,
    evidence_files: Iterable[str | Path] | None = None,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
) -> FinalPipelineResult:
    """Run CR gates, specialist agents, CR-scoped document validation, then final reasoning.

    Level 1: deterministic CR structure/policy checks.
    Level 2: deterministic specialist/context intelligence plus local evidence extraction.
    Level 3: field-aware document content verification and final reasoning.

    The model can add conservative findings but cannot override a deterministic blocker.
    """
    canonical_cr = normalize_cr_record(cr)
    number = str(
        canonical_cr.get("Number")
        or canonical_cr.get("Effective number")
        or canonical_cr.get("change_number")
        or "UNKNOWN_CR"
    ).strip()

    # Level 1/2: always run the deterministic agents first.
    stage1_result = run_stage1(canonical_cr, strictness=strictness, model=None, memory=memory)
    collected = list(documents or [])
    classifications: list[dict[str, Any]] = []
    workspace_root: Path | None = None

    if attachment_root is not None:
        workspace_root = Path(attachment_root).expanduser()
        if evidence_files:
            stage_folder = stage_files(workspace_root, number, evidence_files)
            loaded, classifications = load_and_classify(workspace_root, number)
            collected.extend(loaded)
        else:
            stage_folder = safe_cr_dir(workspace_root, number)
            loaded, classifications = load_and_classify(workspace_root, number)
            collected.extend(loaded)
        stage1_result.stage1.metadata["evidence_workspace"] = {
            "root": str(workspace_root),
            "cr_folder": str(stage_folder),
            "created_or_resolved": True,
            "files_supplied": len(list(evidence_files or [])),
        }

    # Deduplicate explicit and discovered evidence.
    deduped: list[EvidenceDocument] = []
    seen: set[str] = set()
    for document in collected:
        if document.ref in seen:
            continue
        seen.add(document.ref)
        deduped.append(document)

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

    # A document/evidence blocker can never be upgraded by AI reasoning.
    if stage2 is not None:
        stage1_result.stage1.findings.extend(stage2.findings)
        if stage2.decision == Decision.NOT_READY:
            final_decision = Decision.NOT_READY
        elif stage2.decision == Decision.CONDITIONAL and final_decision == Decision.PASS:
            final_decision = Decision.CONDITIONAL

    stage1_result.stage1.metadata.update(
        {
            "model": getattr(model, "model_name", None),
            "model_prediction": reasoning.model_prediction.value if reasoning.model_prediction else None,
            "model_error": reasoning.error,
            "reasoning_mode": reasoning.reasoning.mode if reasoning.reasoning else "none",
            "final_reasoning_model": getattr(model, "model_name", None),
            "final_reasoning_mode": reasoning.reasoning.mode if reasoning.reasoning else "none",
            "documents_analyzed": len(deduped),
            "deterministic_final": deterministic_final.value,
            "final_decision": final_decision.value,
            "brain": payload,
            "evidence_manifest": _evidence_manifest(number, deduped, classifications),
            "decision_levels": {
                "level_1_cr_gates": stage1_result.stage1.decision.value,
                "level_2_evidence_gates": stage2.decision.value if stage2 else "NOT_RUN",
                "level_3_final_reasoning": reasoning.model_prediction.value if reasoning.model_prediction else "NOT_RUN",
                "final": final_decision.value,
            },
        }
    )

    return FinalPipelineResult(
        stage1=stage1_result.stage1,
        stage2=stage2,
        final_decision=final_decision,
        documents=deduped,
        final_reasoning=reasoning,
        agent_results=stage1_result.agent_results,
        evidence_manifest=_evidence_manifest(number, deduped, classifications),
    )
