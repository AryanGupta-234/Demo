"""One-command Pre-CAB brain runner for manual or bulk CR testing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.env_loader import load_dotenv
from pre_cab.input_loader import load_cr_records, normalize_cr_record, record_type, source_id
from pre_cab.narrative import format_agent_chains, format_cab_result
from pre_cab.pipeline import run_pre_cab
from pre_cab.persistent_memory import SQLiteUnifiedMemory
from pre_cab.reporting import build_report
from pre_cab.runtime_provider import build_runtime_provider
from pre_cab.schemas import Strictness


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="One-command Pre-CAB validation for one or many CRs")
    parser.add_argument("cr_json", type=Path)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface", "ollama"], default=None)
    parser.add_argument("--attachment-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pre_cab"))
    parser.add_argument(
        "--include-emergency", action="store_true",
        help="Also process Emergency-type changes (excluded by default - this system is scoped to Normal CAB review).",
    )
    args = parser.parse_args()

    records = load_cr_records(args.cr_json)
    if not records:
        raise SystemExit("Input JSON contains no CR records")

    # Default scope is Normal changes only: Emergency changes don't go through the
    # CAB call this system supports, and scoring them against Normal-change field/
    # historical patterns would be a category error, not a finding. Filtering here
    # (main.py, the one-command entry point) closes a real gap: only batch.py's
    # bulk path had this filter before, so pointing main.py directly at a raw
    # export previously meant burning LLM calls (and CPU time) on out-of-scope
    # records with no way to opt out short of pre-filtering the file yourself.
    if not args.include_emergency:
        before = len(records)
        records = [r for r in records if record_type(r) != "emergency"]
        excluded = before - len(records)
        if excluded:
            print(f"Excluded {excluded} Emergency-type record(s) (use --include-emergency to process them too).")
    if not records:
        raise SystemExit("No in-scope records to process after filtering (all were Emergency-type).")

    # GPT-OSS is the generative reasoning layer. V1 intentionally uses Groq first;
    # Hugging Face remains available as an explicit/automatic fallback. It is
    # never a prerequisite for the deterministic engine, though: the one-command
    # tool must still work with zero setup (see run_pre_cab/run_stage1 - the
    # deterministic safety reconciliation is the final authority regardless of
    # whether a model ran at all). A missing/failing provider degrades to
    # deterministic-only validation, unless the user explicitly named one with
    # --provider, in which case failing to honor that choice silently would be
    # more confusing than just saying so.
    model = None
    have_credentials = any(os.getenv(name) for name in ("GROQ_API_KEY", "HF_TOKEN"))
    if args.provider or have_credentials:
        try:
            model = build_runtime_provider(args.provider)
            print(f"LLM provider/model: {getattr(model, 'model_name', type(model).__name__)}")
        except Exception as exc:
            if args.provider:
                raise SystemExit(f"Could not initialize GPT-OSS 120B provider: {type(exc).__name__}: {exc}") from exc
            print(f"LLM unavailable: {type(exc).__name__}: {exc}; continuing with deterministic evaluation only.")
    else:
        print("No GPT-OSS credentials detected; continuing with deterministic validation only.")

    memory = SQLiteUnifiedMemory(Path(".pre_cab") / "memory.sqlite3")
    audit = SQLiteAuditStore(Path(".pre_cab") / "audit.sqlite3")
    reports: list[dict[str, Any]] = []

    for index, cr in enumerate(records, 1):
        number = source_id(cr)
        print(f"[{index}/{len(records)}] Processing {number}...")
        try:
            result = run_pre_cab(
                cr,
                strictness=Strictness(args.strictness),
                model=model,
                memory=memory,
                attachment_root=args.attachment_root,
            )
            report = build_report(result.stage1, stage2=result.stage2)
            report.update({
                "cr_number": number,
                "final_decision": result.final_decision.value,
                "documents_analyzed": len(result.documents),
                "final_reasoning": result.stage1.metadata.get("brain", {}),
                "llm_model": getattr(model, "model_name", None),
            })
            canonical_cr = normalize_cr_record(cr)
            report["agent_chains_text"] = format_agent_chains(result.agent_results)
            report["cab_block_text"] = format_cab_result(
                number,
                canonical_cr,
                result.final_decision,
                result.stage1.confidence,
                result.agent_results,
                result.stage1.findings,
                result.stage1.metadata.get("brain") or {},
            )
            run_id = audit.record(
                cr_number=number,
                strictness=args.strictness,
                stage1_decision=result.stage1.decision.value,
                stage2_decision=result.stage2.decision.value if result.stage2 else None,
                final_decision=result.final_decision.value,
                model=getattr(model, "model_name", None),
                payload=report,
            )
            report["run_id"] = run_id
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / f"{number}_pre_cab.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            reports.append(report)
        except Exception as exc:
            report = {
                "cr_number": number,
                "final_decision": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            reports.append(report)
            print(f"  ERROR: {report['error']}")

    if len(reports) == 1:
        report = reports[0]
        agent_chains = report.get("agent_chains_text")
        if agent_chains:
            print(f"\n{agent_chains}\n")
        cab_block = report.get("cab_block_text")
        if cab_block:
            print(cab_block)
        else:
            print(f"\nCR {report.get('cr_number')}: {report.get('final_decision')} (see error above)")
    else:
        counts = {"PASS": 0, "CONDITIONAL": 0, "NOT_READY": 0, "ERROR": 0}
        for report in reports:
            decision = str(report.get("final_decision", "ERROR"))
            counts[decision] = counts.get(decision, 0) + 1
        print(
            f"\nPRE-CAB BATCH: {len(reports)} CRs | "
            f"PASS={counts['PASS']} CONDITIONAL={counts['CONDITIONAL']} "
            f"NOT_READY={counts['NOT_READY']} ERROR={counts['ERROR']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_results.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0 if all(r.get("final_decision") != "ERROR" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
