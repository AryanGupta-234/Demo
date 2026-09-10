from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pre_cab.field_intelligence import profile_fields


def _records_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "records", "changes", "data"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value
    raise SystemExit(
        "Input JSON must contain a list of records or an object wrapping one under "
        "result/records/changes/data"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/field_profile.json"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = _records_from_json(payload)
    profiles = profile_fields(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([p.__dict__ for p in profiles], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Profiled {len(profiles)} fields across {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
