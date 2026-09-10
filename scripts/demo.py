#!/usr/bin/env python3
"""Run the synthetic Normal-CR demo without requiring an LLM or network."""
from __future__ import annotations

import json
from pathlib import Path

from pre_cab.evidence import EvidenceDocument, verify_attachments
from pre_cab.reporting import build_report
from pre_cab.schemas import Strictness
from pre_cab.decision import validate_fields

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo" / "normal_crs.json"
ATTACHMENTS = ROOT / "data" / "demo" / "attachments"


def main() -> int:
    records = json.loads(DATA.read_text(encoding="utf-8"))
    for cr in records:
        stage1 = validate_fields(cr, Strictness.BALANCED)
        docs = []
        attachment = ATTACHMENTS / {
            "CHG-DEMO-001": "CHG-DEMO-001_test_report.txt",
            "CHG-DEMO-002": "CHG-DEMO-002_uat_report.txt",
            "CHG-DEMO-003": "CHG-DEMO-003_evidence_mismatch.txt",
        }.get(cr["Number"], "")
        if attachment.exists():
            docs.append(EvidenceDocument(str(attachment), attachment.name, attachment.read_text(encoding="utf-8"), "txt"))
        stage2 = verify_attachments(cr, docs, requirements=stage1.requirements, strictness=Strictness.BALANCED)
        report = build_report(stage1, stage2=stage2)
        print(f"\n{cr['Number']} — {report['decision']}")
        for item in report["cab_view"]["attention_items"]:
            print(f"  - {item['title']}: {item['message']}")
        print(f"  Stage 2: {stage2.decision.value}; verified={stage2.verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
