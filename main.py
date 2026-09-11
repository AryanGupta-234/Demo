"""One-command Pre-CAB brain runner for manual or bulk CR testing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.env_loader import load_dotenv
from pre_cab.input_loader import load_cr_records, source_id
from pre_cab.pipeline import run_pre_cab
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.reporting import build_report
from pre_cab.runtime_provider import build_runtime_provider
from pre_cab.schemas import Strictness


def _print_value(label: str, value: Any) -> None:
    if value in (None, "", [], {}):
        return
    print(f"\n{label}:")
    if isinstance(value, list):
        for item in value:
            print(f"• {item}")
    else:
        print(str(value))


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="One-command Pre-CAB validation for one or many CRs")
    parser.add_argument("cr_json", type=Path)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--provider", choices=["auto", "cerebras", "groq", "huggingface"], default=None)
    parser.add_argument("--attachment-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pre_cab"))
    args = parser.parse_args()

    records = load_cr_records(args.cr_json)
    if not records:
        raise SystemExit("Input JSON contains no CR records")
    if not any(os.getenv(name) for name in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "HF_TOKEN")):
        raise SystemExit("GPT-OSS 120B credentials not detected after loading .env. Set a provider key in .env or the shell.")

    try:
        model = build_runtime_provider(args.provider)
    except Exception as exc:
        raise SystemExit(f"Could not initialize GPT-OSS 120B provider: {type(exc).__name__}: {exc}") from exc

    print(f"LLM provider/model: {getattr(model, 'model_name', type(model).__name__)}")
    memory = SQLiteUnifiedMemory(Path(".pre_cab") / "memory.sqlite3")
    audit = SQLiteAuditStore(Path(".pre_cab") / "audit.sqlite3")
    reports: list[dict[str, Any]] = []

    for index, cr in enumerate(records, 1):
        number = source_id(cr)
        print(f"[{index}/{len(records)}] Processing {number}...")
        try:
            result = run_pre_cab(cr, strictness=Strictness(args.strictness), model=model, memory=memory, attachment_root=args.attachment_root)
            report = build_report(result.stage1, stage2=result.stage2)
            report.update({"cr_number": number, "final_decision": result.final_decision.value, "documents_analyzed": len(result.documents), "final_reasoning": result.stage1.metadata.get("brain", {}), "llm_model": getattr(model, "model_name", None)})
            run_id = audit.record(cr_number=number, strictness=args.strictness, stage1_decision=result.stage1.decision.value, stage2_decision=result.stage2.decision.value if result.stage2 else None, final_decision=result.final_decision.value, model=getattr(model, "model_name", None), payload=report)
            report["run_id"] = run_id
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / f"{number}_pre_cab.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            reports.append(report)
        except Exception as exc:
            report = {"cr_number": number, "final_decision": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            reports.append(report)
            print(f"  ERROR: {report['error']}")

    if len(reports) == 1:
        report = reports[0]
        print("\nPRE-CAB RESULT")
        print("=" * 80)
        print(f"CR: {report.get('cr_number')}")
        print(f"Prediction: {report.get('final_decision')}")
        print(f"Confidence: {float(report.get('confidence', 0)):.0%}")
        cab = report.get("cab_view", {})
        _print_value("CAB SUMMARY", cab.get("summary"))
        _print_value("CAB ATTENTION", cab.get("attention_items"))
        technical = report.get("technical_view", {})
        _print_value("TECHNICAL SUMMARY", technical.get("summary"))
        _print_value("HISTORICAL / CLONE MATCHES", technical.get("historical_matches"))
        brain = report.get("final_reasoning", {})
        for heading, key in (("FACTS", "facts"), ("INFERENCES", "inferences"), ("UNCERTAINTIES", "uncertainties"), ("CONTRADICTIONS", "contradictions"), ("AI TECHNICAL REASONING", "technical_reasoning"), ("AI CAB REASONING", "cab_reasoning"), ("CAB QUESTIONS", "cab_questions"), ("RECOMMENDATIONS", "recommendations"), ("SELF-CRITIQUE", "self_critique")):
            _print_value(heading, brain.get(key))
        print("=" * 80)
    else:
        counts = {"PASS": 0, "CONDITIONAL": 0, "NOT_READY": 0, "ERROR": 0}
        for report in reports:
            decision = str(report.get("final_decision", "ERROR"))
            counts[decision] = counts.get(decision, 0) + 1
        print(f"\nPRE-CAB BATCH: {len(reports)} CRs | PASS={counts['PASS']} CONDITIONAL={counts['CONDITIONAL']} NOT_READY={counts['NOT_READY']} ERROR={counts['ERROR']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_results.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0 if all(r.get("final_decision") != "ERROR" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
