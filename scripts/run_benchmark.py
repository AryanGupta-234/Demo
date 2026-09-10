from __future__ import annotations
import argparse, json, random
from pathlib import Path
from pre_cab.cab_outcomes import normalize_cab_recommendation
from pre_cab.models import GroqGPTOSS120B
from pre_cab.schemas import Strictness, Decision
from pre_cab.orchestrator import run_stage1


def stratified(records, limit, seed=7):
    buckets = {d: [] for d in Decision}
    unknown = []
    for row in records:
        label = normalize_cab_recommendation(row.get("CAB recommendation"))
        (buckets[label] if label else unknown).append(row)
    rng = random.Random(seed)
    selected = []
    usable = [b for b in buckets.values() if b]
    while len(selected) < min(limit, sum(map(len, usable))) and usable:
        for bucket in usable:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(rng.randrange(len(bucket))))
    return selected


def heuristic_predict(cr, strictness):
    return run_stage1(cr, strictness=strictness).stage1.decision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    p.add_argument("--llm", action="store_true", help="Use configured GPT-OSS 120B provider")
    args = p.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    rows = stratified([r for r in records if str(r.get("Type") or "").lower() == "normal"], args.limit)
    model = GroqGPTOSS120B() if args.llm else None
    summary = {"total": len(rows), "scored": 0, "correct": 0, "false_pass": 0}
    for row in rows:
        actual = normalize_cab_recommendation(row.get("CAB recommendation"))
        if actual is None:
            continue
        predicted = run_stage1(row, strictness=Strictness(args.strictness), model=model).stage1.decision
        summary["scored"] += 1
        summary["correct"] += predicted == actual
        summary["false_pass"] += predicted == Decision.PASS and actual != Decision.PASS
    summary["accuracy"] = summary["correct"] / summary["scored"] if summary["scored"] else 0.0
    summary["false_pass_rate"] = summary["false_pass"] / summary["scored"] if summary["scored"] else 0.0
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
