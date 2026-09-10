from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.field_signal_mining import mine_field_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine advisory field/outcome signals from Normal CR history")
    parser.add_argument("input", type=Path)
    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/field_signals.json"))
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON list")

    signals = mine_field_signals(records, minimum_rows=args.minimum_rows)
    payload = [signal.to_dict() for signal in signals[: args.top if args.top > 0 else None]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(payload)} advisory field signals to {args.output}")


if __name__ == "__main__":
    main()
