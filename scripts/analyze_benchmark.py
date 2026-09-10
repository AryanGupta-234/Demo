from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.benchmark_analysis import failure_buckets
from pre_cab.benchmark_runner import BenchmarkPrediction
from pre_cab.schemas import Decision, Strictness


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark safety failures")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    strictness = Strictness(payload["strictness"])
    predictions = [
        BenchmarkPrediction(
            source_id=row["source_id"],
            actual=Decision(row["actual"]),
            predicted=Decision(row["predicted"]),
            confidence=float(row["confidence"]),
            strictness=strictness,
            finding_codes=tuple(row.get("finding_codes", [])),
        )
        for row in payload.get("predictions", [])
    ]
    print(json.dumps(failure_buckets(predictions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
