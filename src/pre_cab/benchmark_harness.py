"""End-to-end benchmark harness with leakage checks, manifests and JSONL output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .benchmark_leakage import POST_DECISION_FIELDS, leakage_report
from .benchmark_manifest import build_manifest
from .benchmark_prepare import BenchmarkExample, prepare_normal_benchmark
from .benchmark_runner import BenchmarkReport, run_benchmark
from .input_loader import load_cr_records
from .schemas import Strictness


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_benchmark_artifacts(
    input_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[BenchmarkExample], dict[str, Any]]:
    """Create sanitized input, private labels, and a reproducibility manifest locally."""
    source = Path(input_path)
    output = Path(output_dir)
    records = load_cr_records(source)
    examples = prepare_normal_benchmark(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "normal_inputs.json").write_text(
        json.dumps([item.input_record for item in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "ground_truth.private.json").write_text(
        json.dumps({item.source_id: item.actual.value if item.actual else None for item in examples}, indent=2),
        encoding="utf-8",
    )
    manifest = build_manifest(
        records,
        seed=7,
        strictness=Strictness.BALANCED.value,
        limit=0,
        provider="none",
        llm_enabled=False,
        mode="prepared",
    )
    manifest["source_file_sha256"] = _sha256_file(source)
    manifest["leakage_report"] = leakage_report(records)
    manifest["protected_field_count"] = len(POST_DECISION_FIELDS)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return examples, manifest


def write_report(report: BenchmarkReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_predictions_jsonl(predictions: Iterable[Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            payload = {
                "source_id": prediction.source_id,
                "actual": prediction.actual.value,
                "predicted": prediction.predicted.value,
                "confidence": prediction.confidence,
                "correct": prediction.correct,
                "false_pass": prediction.false_pass,
                "false_fail": prediction.false_fail,
                "finding_codes": list(prediction.finding_codes),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_fixed_benchmark(
    records: list[dict[str, Any]],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
) -> BenchmarkReport:
    examples = [item for item in prepare_normal_benchmark(records) if item.actual is not None]
    return run_benchmark(examples, strictness=strictness, model=model, memory=memory)
