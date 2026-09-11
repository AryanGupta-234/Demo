"""One-command Pre-CAB brain runner for manual or bulk CR testing.

Give this program one JSON file containing either one CR or many CRs. It automatically initializes
shared persistent memory, selects the GPT-OSS 120B provider, runs the complete Pre-CAB reasoning
pipeline, writes audit/results, and prints a useful CAB/technical summary.

ServiceNow ingestion remains an upstream boundary. Attachment-folder processing is optional through
--attachment-root so the upstream file manager can be added later without changing the brain.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.input_loader import load_cr_records, source_id
from pre_cab.pipeline import run_pre_cab
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.provider_factory import build_provider
from pre_cab.reporting import build_report
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


def _run_one(
    cr: dict[str, Any],
    *,
    model: Any,
    memory: SQLiteUnifiedMemory,
    audit: SQLiteAuditStore,
    strictness: Strictness,
    attachment_root: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    number = source_id(cr)
    result = run_pre_cab(
        cr,
        strictness=strictness,
        model=model,
        memory=memory,
        attachment_root=attachment_root,
    )
    report = build_report(result.stage1, stage2=result.stage2)
    report.update(
        {
            "cr_number": number,
            "final_decision": result.final_decision.value,
            "documents_analyzed": len(result.documents),
            "final_reasoning": result.stage1.metadata.get("brain", {}),
            "test_mode": "json_pre_cab",
            "service_now_ingestion": "upstream_input",
            "attachment_processing": "enabled" if attachment_root else "disabled",
        }
    )

    run_id = audit.record(
        cr_number=number,
        strictness=strictness.value,
        stage1_decision=result.stage1.decision.value,
        stage2_decision=result.stage2.decision.value if result.stage2 else None,
        final_decision=result.final_decision.value,
        model=getattr(model, "model_name", None),
        payload=report,
    )
    report["run_id"] = run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{number}_pre_cab.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report


def _print_one(report: dict[str, Any]) -> None:
    print("\n" + "=" * 82)
    print("PRE-CAB VALIDATION")
    print("=" * 82)
    print(f"CR:         {report.get('cr_number', 'unknown')}")
    print(f"Decision:   {report.get('final_decision', 'unknown')}")
    print(f"Confidence: {float(report.get('confidence', 0.0)):.0%}")
    print(f"Run ID:     {report.get('run_id', 'unknown')}")

    cab = report.get("cab_view") or {}
    _print_value("CAB SUMMARY", cab.get("summary"))
    _print_value("CAB ATTENTION", cab.get("attention_items"))
    _print_value("REQUIREMENTS", cab.get("requirements"))
    _print_value("EVIDENCE", cab.get("evidence"))

    technical = report.get("technical_view") or {}
    _print_value("TECHNICAL SUMMARY", technical.get("summary"))
    _print_value("HISTORICAL / CLONE MATCHES", technical.get("historical_matches"))

    brain = report.get("final_reasoning") or {}
    for heading, key in (
        ("AI TECHNICAL REASONING", "technical_reasoning"),
        ("AI CAB REASONING", "cab_reasoning"),
        ("UNCERTAINTIES", "uncertainties"),
        ("CONTRADICTIONS", "contradictions"),
        ("CAB QUESTIONS", "cab_questions"),
        ("RECOMMENDATIONS", "recommendations"),
        ("SELF-CRITIQUE", "self_critique"),
    ):
        _print_value(heading, brain.get(key))
    print("=" * 82)


def _print_batch_summary(reports: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {"PASS": 0, "CONDITIONAL": 0, "NOT_READY": 0, "ERROR": 0}
    for report in reports:
        decision = str(report.get("final_decision") or "ERROR")
        counts[decision] = counts.get(decision, 0) + 1

    print("\n" + "=" * 82)
    print("PRE-CAB BATCH SUMMARY")
    print("=" * 82)
    print(f"Total CRs:    {len(reports)}")
    print(f"PASS:         {counts.get('PASS', 0)}")
    print(f"CONDITIONAL:  {counts.get('CONDITIONAL', 0)}")
    print(f"NOT_READY:    {counts.get('NOT_READY', 0)}")
    print(f"ERROR:        {counts.get('ERROR', 0)}")
    print("\nDecisions:")
    for report in reports:
        print(f"• {report.get('cr_number', 'unknown')}: {report.get('final_decision', 'ERROR')}")
    print("=" * 82)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command Pre-CAB validation for one or many CRs")
    parser.add_argument("cr_json", type=Path, help="JSON containing one CR or a batch of CR records")
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument(
        "--attachment-root",
        type=Path,
        default=None,
        help="Optional private CR workspace; each <CR number>/ folder is processed automatically",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pre_cab"),
        help="Directory for per-CR JSON results and batch_results.json",
    )
    args = parser.parse_args()

    try:
        records = load_cr_records(args.cr_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not load CR JSON: {exc}") from exc

    if not records:
        raise SystemExit("Input JSON contains no CR records")

    strictness = Strictness(args.strictness)
    if not (os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")):
        raise SystemExit(
            "GPT-OSS 120B credentials not detected. Set GROQ_API_KEY or HF_TOKEN; "
            "the main runner will not silently fall back to deterministic-only mode."
        )

    try:
        model = build_provider(args.provider)
    except Exception as exc:
        raise SystemExit(f"Could not initialize GPT-OSS 120B provider: {type(exc).__name__}: {exc}") from exc

    print(f"LLM provider: {getattr(model, 'model_name', type(model).__name__)}")
    memory = SQLiteUnifiedMemory(Path(".pre_cab") / "memory.sqlite3")
    audit = SQLiteAuditStore(Path(".pre_cab") / "audit.sqlite3")

    reports: list[dict[str, Any]] = []
    for index, cr in enumerate(records, start=1):
        number = source_id(cr)
        print(f"[{index}/{len(records)}] Processing {number}...")
        try:
            report = _run_one(
                cr,
                model=model,
                memory=memory,
                audit=audit,
                strictness=strictness,
                attachment_root=args.attachment_root,
                output_dir=args.output_dir,
            )
        except Exception as exc:
            report = {
                "cr_number": number,
                "final_decision": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "test_mode": "json_pre_cab",
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / f"{number}_pre_cab_error.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  ERROR: {report['error']}")
        reports.append(report)

    if len(reports) == 1:
        _print_one(reports[0])
    else:
        _print_batch_summary(reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = args.output_dir / "batch_results.json"
    batch_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nResults: {batch_path}")
    return 0 if all(report.get("final_decision") != "ERROR" for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
