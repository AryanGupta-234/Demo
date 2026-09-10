from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.feedback import CABFeedback, remember_feedback
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.schemas import Decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Record actual CAB outcome and reviewer corrections")
    parser.add_argument("cr_number")
    parser.add_argument("predicted", choices=[d.value for d in Decision])
    parser.add_argument("actual", choices=[d.value for d in Decision])
    parser.add_argument("--notes", default="")
    parser.add_argument("--requirements", type=Path, default=None, help="JSON object of corrected requirement booleans")
    parser.add_argument("--lesson", action="append", default=[])
    parser.add_argument("--memory-db", type=Path, default=Path(".pre_cab/history.sqlite3"))
    args = parser.parse_args()

    corrected = {}
    if args.requirements:
        payload = json.loads(args.requirements.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("--requirements must point to a JSON object")
        corrected = {str(name): bool(value) for name, value in payload.items()}

    feedback = CABFeedback(
        cr_number=args.cr_number,
        predicted=Decision(args.predicted),
        actual=Decision(args.actual),
        reviewer_notes=args.notes,
        corrected_requirements=corrected,
        lessons=tuple(args.lesson),
    )
    memory = SQLiteUnifiedMemory(args.memory_db)
    record = remember_feedback(memory, feedback)
    print(json.dumps({"memory_id": record.memory_id, "feedback": feedback.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
