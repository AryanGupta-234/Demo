"""Ablation runner for measuring which validator layers add real value."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .benchmark_metrics import compute_safety_metrics
from .benchmark_prepare import BenchmarkExample
from .orchestrator import run_stage1
from .pipeline import run_pre_cab
from .schemas import Decision, Strictness


def _metrics(rows: list[tuple[Decision, Decision]]) -> dict[str, Any]:
    return asdict(compute_safety_metrics(rows))


def run_ablation(
    examples: Iterable[BenchmarkExample],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
    attachment_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare deterministic, memory/clone, evidence and full GPT-OSS strategies."""
    examples = [e for e in examples if e.actual is not None]
    strategies: dict[str, list[dict[str, Any]]] = {
        "deterministic": [],
        "deterministic_memory_clone": [],
        "deterministic_evidence": [],
        "full_gpt_oss": [],
    }
    for example in examples:
        base = run_stage1(example.input_record, strictness=strictness, model=None, memory=None).stage1
        strategies["deterministic"].append({"source_id": example.source_id, "actual": example.actual.value, "predicted": base.decision.value})
        enriched = run_stage1(example.input_record, strictness=strictness, model=None, memory=memory).stage1
        strategies["deterministic_memory_clone"].append({"source_id": example.source_id, "actual": example.actual.value, "predicted": enriched.decision.value})
        evidence_result = run_pre_cab(example.input_record, attachment_root=attachment_root, strictness=strictness, model=None, memory=memory)
        strategies["deterministic_evidence"].append({"source_id": example.source_id, "actual": example.actual.value, "predicted": evidence_result.final_decision.value})
        if model is not None:
            full = run_pre_cab(example.input_record, attachment_root=attachment_root, strictness=strictness, model=model, memory=memory)
            full_prediction = full.final_decision
        else:
            full_prediction = evidence_result.final_decision
        strategies["full_gpt_oss"].append({"source_id": example.source_id, "actual": example.actual.value, "predicted": full_prediction.value})

    output: dict[str, Any] = {"strictness": strictness.value, "count": len(examples), "strategies": {}}
    for name, rows in strategies.items():
        pairs = [(Decision(row["predicted"]), Decision(row["actual"])) for row in rows]
        output["strategies"][name] = {"metrics": _metrics(pairs), "predictions": rows}
    return output
