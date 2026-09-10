"""Prepare a leakage-safe Normal-CR benchmark from a local ServiceNow export.

The source file is never uploaded by this script. It reads locally, keeps only Normal CRs,
extracts the historical outcome into a benchmark-only label, and removes outcome-bearing fields
from the model input record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTCOME_FIELDS = {
    "CAB Outcome",
    "CAB recommendation",
    "CAB Recommendation",
    "Approval history",
}

EMERGENCY_VALUES = {"emergency", "emergency change", "break fix"}
NORMAL_VALUES = {"normal"}


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_outcome(record: dict[str, Any]) -> str | None:
    """Map explicit historical CAB labels to the benchmark decision vocabulary.

    This deliberately refuses to guess from arbitrary comments. Reviewers can extend the mapping
    with organization-specific labels after inspecting the source distribution.
    """
    raw = _clean_text(record.get("CAB Outcome") or record.get("CAB recommendation")).lower()
    if not raw:
        return None
    if any(x in raw for x in ("approved", "approve", "accepted", "proceed")):
        return "PASS"
    if any(x in raw for x in ("conditional", "condition", "hold", "defer", "clarification")):
        return "CONDITIONAL"
    if any(x in raw for x in ("reject", "rejected", "not approved", "not ready")):
        return "NOT_READY"
    return None


def prepare(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        change_type = _clean_text(record.get("Type")).lower()
        if change_type not in NORMAL_VALUES or change_type in EMERGENCY_VALUES:
            continue
        actual = normalize_outcome(record)
        if actual is None:
            continue
        model_record = {k: v for k, v in record.items() if k not in OUTCOME_FIELDS}
        output.append(
            {
                "input": model_record,
                "label": actual,
                "benchmark_id": _clean_text(record.get("Number") or record.get("Effective number")),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit("Expected a JSON array of CR records")

    prepared = prepare(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(prepared)} Normal CR benchmark cases -> {args.output}")


if __name__ == "__main__":
    main()
