"""Summarize a leakage-safe benchmark result file.

Expected rows contain: source_id, actual, predicted, strictness, and optional requirement metrics.
No CR content is printed by this script, only aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Results JSON must contain a list")

    scored = [r for r in rows if r.get("actual") in {"PASS", "CONDITIONAL", "NOT_READY"}]
    pairs = Counter((r.get("actual"), r.get("predicted")) for r in scored)
    correct = sum(1 for r in scored if r.get("actual") == r.get("predicted"))
    false_pass = sum(1 for r in scored if r.get("predicted") == "PASS" and r.get("actual") != "PASS")
    false_fail = sum(1 for r in scored if r.get("predicted") == "NOT_READY" and r.get("actual") == "PASS")

    by_strictness: dict[str, dict[str, Any]] = {}
    for level in sorted({r.get("strictness", "unknown") for r in scored}):
        subset = [r for r in scored if r.get("strictness") == level]
        by_strictness[level] = {
            "n": len(subset),
            "accuracy": sum(r.get("actual") == r.get("predicted") for r in subset) / len(subset) if subset else 0.0,
            "false_pass_rate": sum(r.get("predicted") == "PASS" and r.get("actual") != "PASS" for r in subset) / len(subset) if subset else 0.0,
        }

    output = {
        "total_rows": len(rows),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "accuracy": correct / len(scored) if scored else 0.0,
        "false_pass": false_pass,
        "false_pass_rate": false_pass / len(scored) if scored else 0.0,
        "false_fail": false_fail,
        "confusion": {f"{a}->{p}": n for (a, p), n in sorted(pairs.items())},
        "by_strictness": by_strictness,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
