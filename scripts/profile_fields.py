from __future__ import annotations
import argparse
import json
from pathlib import Path
from pre_cab.field_intelligence import profile_fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/field_profile.json"))
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input JSON must contain a list")
    profiles = profile_fields(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([p.__dict__ for p in profiles], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profiled {len(profiles)} fields across {len(records)} records")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
