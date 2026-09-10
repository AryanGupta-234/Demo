from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.benchmark_harness import prepare_benchmark_artifacts, run_fixed_benchmark, write_predictions_jsonl, write_report
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
    parser.add_argument(
        "--attachment-root",
        type=Path,
        default=None,
        help="Private local evidence directory; enables Stage-2 evidence verification in the benchmark",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Benchmark only deterministic/Stage-1 reasoning instead of the full evidence-aware pipeline",
    )
    args = parser.parse_args()

    examples, manifest = prepare_benchmark_artifacts(args.input, args.output_dir)
    records = load_cr_records(args.input)
    model = build_provider(args.provider) if args.llm else None
    full_pipeline = not args.stage1_only
    report = run_fixed_benchmark(
        records,
        strictness=Strictness(args.strictness),
        model=model,
        full_pipeline=full_pipeline,
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
