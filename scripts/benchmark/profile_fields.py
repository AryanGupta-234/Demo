"""Profile ServiceNow CR fields from a local export.

The profiler is descriptive: it does not silently turn population frequency into policy.
It produces evidence that can be reviewed when building the organization's validation matrix.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IGNORED_TYPES = {"emergency", "emergency change", "break fix"}


def norm(v: Any) -> str:
    return "" if v is None else str(v).strip()


def profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [r for r in records if norm(r.get("Type")).lower() == "normal"]
    fields = sorted({k for r in normal for k in r})
    result: dict[str, Any] = {"records": len(normal), "fields": {}}

    for field in fields:
        values = [norm(r.get(field)) for r in normal]
        populated = [v for v in values if v]
        value_counts = Counter(v for v in populated)
        result["fields"][field] = {
            "populated": len(populated),
            "population_rate": round(len(populated) / len(normal), 4) if normal else 0.0,
            "distinct_values": len(value_counts),
            "top_values": value_counts.most_common(10),
        }

    # Dependency evidence: how often one field is populated when another is.
    dependencies: list[dict[str, Any]] = []
    for a in fields:
        for b in fields:
            if a >= b:
                continue
            a_rows = [r for r in normal if norm(r.get(a))]
            if len(a_rows) < max(10, int(len(normal) * 0.05)):
                continue
            conditional = sum(1 for r in a_rows if norm(r.get(b))) / len(a_rows)
            if conditional >= 0.90 and len(a_rows) >= 25:
                dependencies.append({"when_populated": a, "often_populated": b, "rate": round(conditional, 4)})

    result["dependencies"] = dependencies[:500]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Expected a JSON array")
    result = profile(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profiled {result['records']} Normal CRs and {len(result['fields'])} fields")


if __name__ == "__main__":
    main()
