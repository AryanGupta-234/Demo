from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pre_cab.benchmark_prepare import prepare_normal_benchmark
from pre_cab.benchmark_runner import run_benchmark
from pre_cab.benchmark_split import split_records
from pre_cab.embeddings import LocalSentenceTransformer
from pre_cab.history_memory import build_history_memory
from pre_cab.provider_factory import build_provider
from pre_cab.schemas import Strictness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a leakage-safe holdout Normal-CR benchmark")
    parser.add_argument("input", type=Path)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="Limit unseen test examples; 0 means all")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--embeddings", action="store_true", help="Enable local semantic history retrieval")
    parser.add_argument("--output", type=Path, default=Path("artifacts/holdout_benchmark.json"))
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON list")

    train, validation, test = split_records(records, seed=args.seed)
    reference_rows = train + validation
    embedding_provider = LocalSentenceTransformer() if args.embeddings else None
    memory = build_history_memory(reference_rows, embedding_provider=embedding_provider)

    test_examples = [example for example in prepare_normal_benchmark(test) if example.actual is not None]
    if args.limit > 0 and args.limit < len(test_examples):
        rng = random.Random(args.seed)
        rng.shuffle(test_examples)
        test_examples = test_examples[: args.limit]

    model = build_provider(args.provider) if args.llm else None
    report = run_benchmark(
        test_examples,
        strictness=Strictness(args.strictness),
        model=model,
        memory=memory,
    )
    payload = {
        "split": {
            "reference_train": len(train),
            "reference_validation": len(validation),
            "unseen_test": len(test),
            "scored_test": report.metrics.total_scored,
            "seed": args.seed,
        },
        "retrieval": {
            "semantic_embeddings": bool(args.embeddings),
            "embedding_model": getattr(embedding_provider, "model_name", None),
        },
        "llm": {
            "enabled": bool(args.llm),
            "provider": args.provider if args.llm else None,
            "reasoning_mode": "single by default; set PRE_CAB_REASONING_MODE=dual to override",
        },
        "report": report.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"split": payload["split"], "metrics": payload["report"]["metrics"]}, indent=2))
    print(f"Full holdout report written to {args.output}")


if __name__ == "__main__":
    main()
