from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.requirement_evaluation import RequirementTruth, evaluate_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contextual requirement inference against reviewed truth")
    parser.add_argument("crs", type=Path, help="Local CR JSON list")
    parser.add_argument("truth", type=Path, help="Reviewed JSON list: {cr_id, requirements:{...}}")
    parser.add_argument("--output", type=Path, default=Path("artifacts/requirement_metrics.json"))
    args = parser.parse_args()

    crs = json.loads(args.crs.read_text(encoding="utf-8"))
    truth_payload = json.loads(args.truth.read_text(encoding="utf-8"))
    if not isinstance(crs, list) or not isinstance(truth_payload, list):
        raise SystemExit("Both input files must contain JSON lists")

    truth = [
        RequirementTruth(
            cr_id=str(item["cr_id"]),
            requirements={str(name): bool(value) for name, value in dict(item.get("requirements") or {}).items()},
        )
        for item in truth_payload
        if isinstance(item, dict) and item.get("cr_id")
    ]
    metrics = evaluate_requirements(crs, truth)
    payload = {name: metric.to_dict() for name, metric in metrics.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
