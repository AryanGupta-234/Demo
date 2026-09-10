from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pre_cab.benchmark_prepare import prepare_normal_benchmark
from pre_cab.benchmark_runner import run_benchmark, strictness_sweep
from pre_cab.provider_factory import build_provider
from pre_cab.schemas import Strictness


def _sample(examples, limit: int | None, seed: int):
    examples = [example for example in examples if example.actual is not None]
    if limit is None or limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe historical Normal-CR benchmark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=50, help="0 means all scorable examples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--sweep", action="store_true", help="Evaluate all strictness profiles")
    parser.add_argument("--llm", action="store_true", help="Use configured GPT-OSS 120B provider")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_report.json"))
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON list of ServiceNow CR records")

    examples = _sample(prepare_normal_benchmark(records), args.limit, args.seed)
    model = build_provider(args.provider) if args.llm else None

    if args.sweep:
        reports = strictness_sweep(examples, model=model)
        payload = {name: report.to_dict() for name, report in reports.items()}
        console = {name: report["metrics"] for name, report in payload.items()}
    else:
        report = run_benchmark(examples, strictness=Strictness(args.strictness), model=model)
        payload = report.to_dict()
        console = payload["metrics"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(console, indent=2))
    print(f"Full benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
