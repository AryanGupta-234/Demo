"""Resumable batch validation for large historical/current CR sets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .input_loader import normalize_cr_record, record_type, source_id
from .orchestrator import run_stage1
from .pipeline import run_pre_cab
from .schemas import Strictness


@dataclass(frozen=True)
class BatchProgress:
    processed: int
    skipped: int
    failed: int
    output_path: str


def _cr_id(cr: dict[str, Any]) -> str:
    value = source_id(cr)
    return "" if value.lower() == "unknown" else value


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = str(row.get("cr_number") or "").strip()
            if source_id:
                ids.add(source_id)
    return ids


def run_batch(
    records: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
    attachment_root: str | Path | None = None,
    stage1_only: bool = False,
    normal_only: bool = True,
    resume: bool = True,
) -> BatchProgress:
    """Process CRs sequentially and checkpoint each result as JSONL.

    Sequential execution is intentional for free-tier inference: provider/budget wrappers control
    quota while JSONL checkpointing makes interruption/restart safe.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output) if resume else set()
    mode = "a" if resume else "w"
    processed = skipped = failed = 0

    with output.open(mode, encoding="utf-8") as handle:
        for original in records:
            cr = normalize_cr_record(original)
            number = _cr_id(cr)
            if normal_only and record_type(cr) != "normal":
                skipped += 1
                continue
            if not number or number in done:
                skipped += 1
                continue
            row: dict[str, Any]
            try:
                if stage1_only:
                    validation_result = run_stage1(cr, strictness=strictness, model=model, memory=memory).stage1
                    row = {
                        "cr_number": number,
                        "ok": True,
                        "validation_mode": "metadata_only",
                        "stage1_decision": validation_result.decision.value,
                        "stage2_decision": None,
                        "final_decision": validation_result.decision.value,
                        "confidence": validation_result.confidence,
                        "finding_codes": [finding.code for finding in validation_result.findings],
                        "documents_analyzed": 0,
                        "model_prediction": validation_result.metadata.get("model_prediction"),
                        "model_error": validation_result.metadata.get("model_error"),
                    }
                else:
                    pipeline_result = run_pre_cab(
                        cr,
                        attachment_root=attachment_root,
                        strictness=strictness,
                        model=model,
                        memory=memory,
                    )
                    row = {
                        "cr_number": number,
                        "ok": True,
                        "validation_mode": "evidence_aware",
                        "stage1_decision": pipeline_result.stage1.decision.value,
                        "stage2_decision": pipeline_result.stage2.decision.value if pipeline_result.stage2 else None,
                        "final_decision": pipeline_result.final_decision.value,
                        "confidence": pipeline_result.stage1.confidence,
                        "finding_codes": [finding.code for finding in pipeline_result.stage1.findings]
                        + ([finding.code for finding in pipeline_result.stage2.findings] if pipeline_result.stage2 else []),
                        "documents_analyzed": len(pipeline_result.documents),
                        "model_prediction": (
                            pipeline_result.final_reasoning.model_prediction.value
                            if pipeline_result.final_reasoning and pipeline_result.final_reasoning.model_prediction
                            else None
                        ),
                        "model_error": pipeline_result.final_reasoning.error if pipeline_result.final_reasoning else None,
                    }
                processed += 1
            except Exception as exc:  # batch boundary must checkpoint a failure rather than lose the run
                row = {
                    "cr_number": number,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failed += 1
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            # Avoid processing duplicate CR numbers that occur later in one export.
            done.add(number)

    return BatchProgress(processed, skipped, failed, str(output))
