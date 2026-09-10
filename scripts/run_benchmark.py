from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pre_cab.benchmark_leakage import strip_post_decision_fields
from pre_cab.cab_outcomes import normalize_cab_recommendation
from pre_cab.models import GroqGPTOSS120B
from pre_cab.providers import create_provider
from pre_cab.schemas import Strictness, Decision
from pre_cab.orchestrator import run_stage1


def model_visible(row: dict) -> dict:
    return strip_post_decision_fields(row)


def actual_label(row: dict) -> Decision | None:
    return normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))


def stratified(records: list[dict], limit: int, seed: int = 7) -> list[dict]:
    buckets = {d: [] for d in Decision}
    for row in records:
        label = actual_label(row)
        if label is not None:
            buckets[label].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    while len(selected) < min(limit, sum(len(b) for b in buckets.values())):
        progressed = False
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(rng.randrange(len(bucket))))
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    p.add_argument("--llm", action="store_true", help="Use configured GPT-OSS 120B provider")
    args = p.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    rows = stratified([r for r in records if str(r.get("Type") or "").lower() == "normal"], args.limit)

    model = None
    if args.llm:
        model = create_provider()

    summary = {"total": len(rows), "scored": 0, "correct": 0, "false_pass": 0, "false_fail": 0}
    result_rows = []
    for row in rows:
        actual = actual_label(row)
        if actual is None:
            continue
        predicted = run_stage1(model_visible(row), strictness=Strictness(args.strictness), model=model).stage1.decision
        summary["scored"] += 1
        summary["correct"] += int(predicted == actual)
        summary["false_pass"] += int(predicted == Decision.PASS and actual != Decision.PASS)
        summary["false_fail"] += int(predicted == Decision.NOT_READY and actual == Decision.PASS)
        result_rows.append({
            "source_id": row.get("Number") or row.get("Effective number"),
            "actual": actual.value,
            "predicted": predicted.value,
            "strictness": args.strictness,
        })
    summary["accuracy"] = summary["correct"] / summary["scored"] if summary["scored"] else 0.0
    summary["false_pass_rate"] = summary["false_pass"] / summary["scored"] if summary["scored"] else 0.0
    summary["false_fail_rate"] = summary["false_fail"] / summary["scored"] if summary["scored"] else 0.0
    print(json.dumps(summary, indent=2))
    if args.llm:
        args.input.parent.joinpath("benchmark_results.json").write_text(
            json.dumps(result_rows, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
