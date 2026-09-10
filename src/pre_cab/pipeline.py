"""Stable public entrypoint for the two-stage Pre-CAB pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import EvidenceDocument, EvidenceResult, verify_attachments
from .local_attachments import load_attachments_for_cr
from .orchestrator import run_stage1
from .schemas import Decision, Strictness


@dataclass
class FinalPipelineResult:
    stage1: Any
    stage2: EvidenceResult | None
    final_decision: Decision
    documents: list[EvidenceDocument]


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
    """Evaluate a Normal CR and independently verify available attachments.

    Explicit ``documents`` and locally discovered attachments can be combined. A Stage-1 NOT_READY
    result stops evidence work by default because the CR already has a blocking field-level issue.
    """
    stage1_result = run_stage1(cr, strictness=strictness, model=model, memory=memory)
    collected = list(documents or [])

    if attachment_root is not None:
        cr_number = str(cr.get("Number") or cr.get("Effective number") or "").strip()
        collected.extend(load_attachments_for_cr(Path(attachment_root), cr_number))

    # De-duplicate by reference while preserving caller order.
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
        )

    stage2 = verify_attachments(
        cr,
        deduped,
        requirements=stage1_result.stage1.requirements,
        strictness=strictness,
    )
    final_decision = _merge_decisions(stage1_result.stage1.decision, stage2.decision)
    return FinalPipelineResult(
        stage1=stage1_result.stage1,
        stage2=stage2,
        final_decision=final_decision,
        documents=deduped,
    )
