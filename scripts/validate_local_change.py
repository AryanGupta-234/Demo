from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.pipeline import run_pre_cab
from pre_cab.provider_factory import build_provider
from pre_cab.reporting import build_report
from pre_cab.schemas import Strictness


def _find_cr(records: list[dict], number: str) -> dict:
    for row in records:
        candidate = str(row.get("Number") or row.get("Effective number") or "").strip()
        if candidate.lower() == number.lower():
            return row
    raise KeyError(f"CR not found: {number}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one local Normal CR")
    parser.add_argument("input", type=Path, help="Local JSON export")
    parser.add_argument("number", help="CR number")
    parser.add_argument("--memory-db", type=Path, default=None)
    parser.add_argument("--attachments", type=Path, default=None)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must contain a JSON list")
    cr = _find_cr(records, args.number)
    memory = SQLiteUnifiedMemory(args.memory_db) if args.memory_db else None
    model = build_provider(args.provider) if args.llm else None
    strictness = Strictness(args.strictness)

    result = run_pre_cab(
        cr,
        attachment_root=args.attachments,
        strictness=strictness,
        model=model,
        memory=memory,
    )
    report = build_report(result.stage1, stage2=result.stage2)
    report["final_decision"] = result.final_decision.value
    report["documents_analyzed"] = len(result.documents)
    report["final_reasoning"] = result.stage1.metadata.get("brain", {})

    audit = SQLiteAuditStore()
    run_id = audit.record(
        cr_number=args.number,
        strictness=strictness.value,
        stage1_decision=result.stage1.decision.value,
        stage2_decision=result.stage2.decision.value if result.stage2 else None,
        final_decision=result.final_decision.value,
        model=getattr(model, "model_name", None),
        payload=report,
    )
    report["run_id"] = run_id

    output = args.output or Path("artifacts") / f"{args.number}_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "final_decision": result.final_decision.value, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
