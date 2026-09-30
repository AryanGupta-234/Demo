"""Validate and automatically promote a mined field-requirement table.

Example:
    python scripts/promote_field_requirements.py \
        config/field_requirements.generated.json \
        private_data/holdout.json \
        --out config/field_requirements.generated.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.rule_lifecycle import automatically_promote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Candidate rule-table JSON.")
    parser.add_argument("holdout", type=Path, help="Separate historical Normal-CR holdout JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output rule-table JSON.")
    parser.add_argument("--min-holdout-records", type=int, default=100)
    parser.add_argument("--required-fill-floor", type=float, default=0.85)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    records = json.loads(args.holdout.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise SystemExit("Candidate rule table must be a JSON object.")
    if not isinstance(records, list):
        raise SystemExit("Holdout input must be a JSON list of CR records.")

    promoted = automatically_promote(
        candidate,
        records,
        min_holdout_records=args.min_holdout_records,
        required_fill_floor=args.required_fill_floor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(promoted, indent=2, ensure_ascii=False), encoding="utf-8")

    validation = promoted.get("validation") or {}
    print(
        f"Lifecycle: {promoted.get('lifecycle_status', 'UNKNOWN')} | "
        f"holdout Normal CRs={validation.get('normal_holdout_record_count', 0)} | "
        f"stability={validation.get('stability_rate', 0):.1%} | "
        f"rules={validation.get('rules_evaluated', 0)} | "
        f"output={args.out}"
    )
    print(f"Reason: {validation.get('promotion_reason', 'not available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
