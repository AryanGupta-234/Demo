from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.pipeline import run_pre_cab
from pre_cab.provider_factory import build_provider
from pre_cab.reporting import build_report
from pre_cab.schemas import Strictness
from pre_cab.servicenow_readonly import ReadOnlyServiceNowClient, ServiceNowCredentials


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one ServiceNow Normal CR using GET-only API access")
    parser.add_argument("number")
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-db", type=Path, default=Path(".pre_cab/audit.sqlite3"))
    args = parser.parse_args()

    base_url = os.getenv("SERVICENOW_BASE_URL")
    username = os.getenv("SERVICENOW_USERNAME")
    password = os.getenv("SERVICENOW_PASSWORD")
    if not all((base_url, username, password)):
        raise SystemExit("Set SERVICENOW_BASE_URL, SERVICENOW_USERNAME, and SERVICENOW_PASSWORD")

    client = ReadOnlyServiceNowClient(ServiceNowCredentials(base_url, username, password))
    cr, documents = client.ingest_change(args.number)
    model = build_provider(args.provider) if args.llm else None
    strictness = Strictness(args.strictness)
    result = run_pre_cab(cr, documents=documents, strictness=strictness, model=model)
    report = build_report(result.stage1, stage2=result.stage2)
    report["final_decision"] = result.final_decision.value
    report["documents_analyzed"] = len(result.documents)

    audit = SQLiteAuditStore(args.audit_db)
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
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "final_decision": result.final_decision.value, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
