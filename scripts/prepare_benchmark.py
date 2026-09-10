"""Prepare a local ServiceNow JSON export for the leakage-safe Normal CR benchmark.

This script reads a local file only; it does not upload data anywhere. It writes a sanitized benchmark
input file plus a private ground-truth file so the model cannot see the historical CAB outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.benchmark_prepare import prepare_normal_benchmark
from pre_cab.input_loader import load_cr_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Local JSON export containing CR records")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args()

    records = load_cr_records(args.input)
    examples = prepare_normal_benchmark(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_input = [example.input_record for example in examples]
    ground_truth = {
        example.source_id: example.actual.value if example.actual else None
        for example in examples
    }

    (args.output_dir / "normal_inputs.json").write_text(
        json.dumps(model_input, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ground_truth.private.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(examples)} Normal CR benchmark examples in {args.output_dir}")
    print("Keep ground_truth.private.json out of model prompts and public repositories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
