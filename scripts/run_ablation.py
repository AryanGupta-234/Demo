from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pre_cab.benchmark_ablation import run_ablation
from pre_cab.benchmark_manifest import build_manifest
from pre_cab.benchmark_prepare import prepare_normal_benchmark
from pre_cab.input_loader import load_cr_records
from pre_cab.provider_factory import build_provider
from pre_cab.schemas import Strictness


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Pre-CAB validation layers on one fixed benchmark sample")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=50, help="0 means all scorable examples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--attachment-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ablation_report.json"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/ablation_manifest.json"))
    args = parser.parse_args()

    records = load_cr_records(args.input)
    examples = [e for e in prepare_normal_benchmark(records) if e.actual is not None]
    if not examples:
        raise SystemExit("No scorable Normal CR examples were found in the supplied export.")
    if args.limit > 0 and args.limit < len(examples):
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        examples = examples[:args.limit]

    model = build_provider(args.provider) if args.llm else None
    report = run_ablation(
        examples,
        strictness=Strictness(args.strictness),
        model=model,
        attachment_root=args.attachment_root,
    )
    manifest = build_manifest(
        records,
        seed=args.seed,
        strictness=args.strictness,
        limit=args.limit,
        provider=args.provider,
        llm_enabled=args.llm,
        mode="ablation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, result in report["strategies"].items():
        print(name, json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
