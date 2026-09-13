from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def priority(prediction: str | None, actual: str | None) -> int:
    if not prediction or not actual:
        return 0
    if prediction == "PASS" and actual != "PASS":
        return 100
    if prediction != actual:
        return 60
    return 10


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the next adaptive correction set from reviewed outcomes")
    parser.add_argument("--feedback", type=Path, required=True, help="JSONL with cr_id, input, prediction, actual_outcome, notes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.feedback.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda r: priority(r.get("prediction"), r.get("actual_outcome")), reverse=True)
    report = {
        "feedback_records": len(rows),
        "outcomes": dict(Counter(str(r.get("actual_outcome")) for r in rows if r.get("actual_outcome"))),
        "high_value_errors": sum(priority(r.get("prediction"), r.get("actual_outcome")) >= 60 for r in rows),
        "false_passes": sum(r.get("prediction") == "PASS" and r.get("actual_outcome") != "PASS" for r in rows),
        "ranked_feedback": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
