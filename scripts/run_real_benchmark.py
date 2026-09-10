from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.benchmark_harness import prepare_benchmark_artifacts, run_fixed_benchmark, write_predictions_jsonl, write_report
from pre_cab.benchmark_prepare import benchmark_diagnostics
from pre_cab.input_loader import load_cr_records
from pre_cab.provider_factory import build_provider
from pre_cab.schemas import Strictness


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full local Normal-CR benchmark safely")
    parser.add_argument("input", type=Path, help="Private local ServiceNow export")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/real_benchmark"))
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--llm", action="store_true", help="Use GPT-OSS 120B for final reasoning")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--attachment-root", type=Path, default=None, help="Private local evidence directory")
    parser.add_argument("--stage1-only", action="store_true", help="Benchmark deterministic Stage 1 only")
    args = parser.parse_args()

    records = load_cr_records(args.input)
    diagnostics = benchmark_diagnostics(records)
    print(json.dumps({"input_diagnostics": diagnostics}, indent=2))
    if diagnostics["normal_records"] == 0:
        raise SystemExit("No Normal CR records detected. Review input field/type normalization before benchmarking.")
    if diagnostics["scorable_records"] == 0:
        raise SystemExit("Normal CRs were detected, but no historical CAB outcomes could be normalized. Review candidate_outcome_values above or supply the correct CAB outcome field.")

    examples, manifest = prepare_benchmark_artifacts(args.input, args.output_dir)
    model = build_provider(args.provider) if args.llm else None
    report = run_fixed_benchmark(
        records,
        strictness=Strictness(args.strictness),
        model=model,
        full_pipeline=not args.stage1_only,
        attachment_root=args.attachment_root,
    )
    write_report(report, args.output_dir / "benchmark_report.json")
    write_predictions_jsonl(report.predictions, args.output_dir / "predictions.jsonl")

    manifest.update({
        "strictness": args.strictness,
        "provider": args.provider,
        "llm_enabled": args.llm,
        "scorable_examples": len(examples),
        "pipeline": "stage1" if args.stage1_only else "full",
        "attachment_root_configured": args.attachment_root is not None,
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict()["metrics"], indent=2))
    print(f"Scored examples: {len(report.predictions)}")
    print(f"Pipeline: {'stage1' if args.stage1_only else 'full evidence-aware'}")
    print(f"Artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
