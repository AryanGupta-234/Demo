"""Stable public entrypoint for the two-stage Pre-CAB pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceDocument, EvidenceResult, verify_attachments
from .orchestrator import run_stage1
from .schemas import Decision, Strictness


@dataclass
class FinalPipelineResult:
    stage1: Any
    stage2: EvidenceResult | None
    final_decision: Decision


def run_pre_cab(
    cr: dict[str, Any],
    documents: list[EvidenceDocument] | None = None,
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
) -> FinalPipelineResult:
    """Evaluate a Normal CR, then verify supplied attachments when Stage 1 permits it."""
    stage1 = run_stage1(cr, strictness=strictness, model=model, memory=memory)
    if stage1.stage1.decision == Decision.NOT_READY:
        return FinalPipelineResult(stage1=stage1.stage1, stage2=None, final_decision=Decision.NOT_READY)

    stage2 = verify_attachments(
        cr,
        documents or [],
        requirements=stage1.stage1.requirements,
        strictness=strictness,
    )
    return FinalPipelineResult(stage1=stage1.stage1, stage2=stage2, final_decision=stage2.decision)
