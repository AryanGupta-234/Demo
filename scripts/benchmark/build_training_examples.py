"""Build reasoning-rich fine-tuning candidates from a prepared benchmark.

This does not train a model. It creates reviewable examples with explicit facts, inferred requirements,
verification state and outcome. Split assignment is deterministic so experiments are reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def split_for(identifier: str) -> str:
    value = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def build(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        cr = dict(record.get("input") or {})
        benchmark_id = str(record.get("benchmark_id") or cr.get("Number") or "unknown")
        examples.append(
            {
                "id": benchmark_id,
                "split": split_for(benchmark_id),
                "input": cr,
                "target": {
                    "decision": record["label"],
                    "reasoning_task": (
                        "Determine applicable requirements, identify evidence gaps or contradictions, "
                        "compare relevant historical patterns, and justify the predicted Pre-CAB outcome."
                    ),
                },
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected prepared benchmark JSON array")
    examples = build(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {split: sum(1 for e in examples if e["split"] == split) for split in ("train", "validation", "test")}
    print(f"Built {len(examples)} examples: {counts}")


if __name__ == "__main__":
    main()
