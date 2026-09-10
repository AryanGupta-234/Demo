"""One-command Pre-CAB brain runner for manual CR testing.

Current test boundary: one manually supplied CR JSON + API-backed GPT-OSS 120B reasoning.
ServiceNow ingestion and attachment-folder processing remain upstream inputs and are intentionally
not required for this test mode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.input_loader import load_cr_records
from pre_cab.pipeline import run_pre_cab
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.provider_factory import build_provider
from pre_cab.reporting import build_report
from pre_cab.schemas import Strictness


def _single_cr(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise SystemExit(
            f"Expected exactly one CR for main testing, found {len(records)}. "
            "Create a one-record JSON file."
        )
    return records[0]


def _print_report(report: dict[str, Any], run_id: str) -> None:
    decision = report["final_decision"]
    confidence = report["confidence"]
    print("\n" + "=" * 78)
    print("PRE-CAB VALIDATION")
    print("=" * 78)
    print(f"CR:         {report.get('cr_number', 'unknown')}")
    print(f"Decision:   {decision}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Run ID:     {run_id}")
    print("\nCAB VIEW")
    print("-" * 78)
    print(report["cab_view"]["summary"] or "No CAB summary returned.")

    attention = report["cab_view"].get("attention_items") or []
    if attention:
        print("\nCAB ATTENTION")
        for item in attention:
            print(f"• {item['title']}: {item['message']}")
            if item.get("action"):
                print(f"  Action: {item['action']}")

    requirements = report["cab_view"].get("requirements") or []
    if requirements:
        print("\nREQUIREMENTS")
        for requirement in requirements:
            if isinstance(requirement, dict):
                print(
                    f"• {requirement.get('name')}: required={requirement.get('required')} "
                    f"confidence={requirement.get('confidence', 0):.0%}"
                )

    evidence = report["cab_view"].get("evidence") or {}
    if evidence:
        print("\nEVIDENCE")
        print(f"• Documents considered: {len(evidence.get('documents_considered') or [])}")
        print(f"• Verified: {evidence.get('verified', False)}")
        contradictions = evidence.get("contradictions") or []
        for contradiction in contradictions:
            print(f"• CONTRADICTION: {contradiction}")

    print("\nTECHNICAL VIEW")
    print("-" * 78)
    print(report["technical_view"]["summary"] or "No technical summary returned.")

    historical = report["technical_view"].get("historical_matches") or []
    if historical:
        print("\nHISTORICAL / CLONE MATCHES")
        for match in historical[:5]:
            print(
                f"• {match.get('change_id', 'unknown')}: "
                f"similarity={float(match.get('similarity', 0)):.0%}; "
                f"outcome={match.get('historical_decision', 'unknown')}"
            )

    brain = report.get("final_reasoning") or {}
    if brain:
        print("\nAI REASONING")
        for heading, key in (
            ("Technical", "technical_reasoning"),
            ("CAB", "cab_reasoning"),
            ("Uncertainties", "uncertainties"),
            ("Contradictions", "contradictions"),
            ("CAB Questions", "cab_questions"),
            ("Recommendations", "recommendations"),
            ("Self-critique", "self_critique"),
        ):
            value = brain.get(key)
            if value:
                print(f"\n{heading}:")
                if isinstance(value, list):
                    for item in value:
                        print(f"• {item}")
                else:
                    print(str(value))
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command manual Pre-CAB GPT-OSS 120B brain test"
    )
    parser.add_argument("cr_json", type=Path, help="JSON file containing exactly one CR")
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    try:
        records = load_cr_records(args.cr_json)
        cr = _single_cr(records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not load CR JSON: {exc}") from exc

    number = str(cr.get("Number") or cr.get("Effective number") or "MANUAL-CR").strip()
    strictness = Strictness(args.strictness)

    if not (os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")):
        raise SystemExit("Set GROQ_API_KEY and/or HF_TOKEN before running the Pre-CAB brain.")

    model = build_provider(args.provider)
    memory = SQLiteUnifiedMemory(Path(".pre_cab") / "memory.sqlite3")
    result = run_pre_cab(
        cr,
        strictness=strictness,
        model=model,
        memory=memory,
        # Main manual test intentionally does not process attachments yet.
        attachment_root=None,
    )

    report = build_report(result.stage1, stage2=result.stage2)
    report["cr_number"] = number
    report["final_decision"] = result.final_decision.value
    report["documents_analyzed"] = 0
    report["final_reasoning"] = result.stage1.metadata.get("brain", {})
    report["test_mode"] = "manual_cr_api_brain"
    report["service_now_ingestion"] = "upstream_future_input"
    report["attachment_processing"] = "disabled_for_current_main_test"

    audit = SQLiteAuditStore(Path(".pre_cab") / "audit.sqlite3")
    run_id = audit.record(
        cr_number=number,
        strictness=strictness.value,
        stage1_decision=result.stage1.decision.value,
        stage2_decision=result.stage2.decision.value if result.stage2 else None,
        final_decision=result.final_decision.value,
        model=getattr(model, "model_name", None),
        payload=report,
    )

    output = args.output or Path("artifacts") / f"{number}_pre_cab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _print_report(report, run_id)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
