"""Offline diagnostic for one Pre-CAB CR and its local evidence workspace."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from pre_cab.decision import validate_fields
from pre_cab.evidence_workspace import load_and_classify, safe_cr_dir
from pre_cab.input_loader import load_cr_records, normalize_cr_record, source_id
from pre_cab.local_attachments import inventory_workspace
from pre_cab.schemas import Decision, Strictness


def _check_package(name: str) -> dict[str, object]:
    try:
        module = __import__(name)
        return {"name": name, "installed": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"name": name, "installed": False, "error": f"{type(exc).__name__}: {exc}"}


def diagnose(cr_path: Path, *, evidence_root: Path | None, strictness: Strictness, index: int = 0) -> dict[str, object]:
    records = load_cr_records(cr_path)
    if not records:
        raise ValueError("No CR records found")
    if index < 0 or index >= len(records):
        raise IndexError(f"CR index {index} is outside 0..{len(records)-1}")

    cr = normalize_cr_record(records[index])
    number = source_id(cr)
    validation = validate_fields(cr, strictness)

    report: dict[str, object] = {
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        },
        "configuration": {
            "strictness": strictness.value,
            "semantic_memory": os.getenv("PRE_CAB_SEMANTIC_MEMORY", "true"),
            "embedding_model": os.getenv("PRE_CAB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "max_attachment_bytes": os.getenv("PRE_CAB_MAX_ATTACHMENT_BYTES", "78643200"),
        },
        "dependencies": [
            _check_package("pypdf"),
            _check_package("openpyxl"),
            _check_package("docx"),
            _check_package("pptx"),
            _check_package("sentence_transformers"),
        ],
        "cr": {
            "number": number,
            "type": cr.get("Type"),
            "category": cr.get("Category"),
            "sub_category": cr.get("Sub Category"),
            "normalized_field_count": len(cr),
        },
        "level_1": {
            "decision": validation.decision.value,
            "confidence": validation.confidence,
            "blocking": [f.code for f in validation.blocking_findings],
            "warnings": [f.code for f in validation.findings if f.severity.value == "WARNING"],
            "requirements": [
                {"name": r.name, "required": r.required, "reason": r.reason}
                for r in validation.requirements
            ],
        },
    }

    if evidence_root is None:
        return report

    root = evidence_root.expanduser().resolve()
    folder = safe_cr_dir(root, number)
    inventory = inventory_workspace(root, number)
    docs, classifications = load_and_classify(root, number)
    report["evidence_workspace"] = {
        "root": str(root),
        "cr_folder": str(folder),
        "inventory_count": len(inventory),
        "supported_count": sum(bool(x.get("supported")) for x in inventory),
        "unsupported_count": sum(not bool(x.get("supported")) for x in inventory),
        "documents_extracted": len(docs),
        "classifications": classifications,
        "inventory": inventory,
        "extraction_errors": [
            {"name": d.name, "error": d.metadata.get("extraction_error")}
            for d in docs if d.metadata.get("extraction_error")
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose one local Pre-CAB CR and evidence workspace")
    parser.add_argument("cr_json", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = parser.parse_args()

    report = diagnose(
        args.cr_json,
        evidence_root=args.evidence_root,
        strictness=Strictness(args.strictness),
        index=args.index,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"CR: {report['cr']['number']} | Type={report['cr']['type']}")
    level1 = report["level_1"]
    print(f"Level 1: {level1['decision']} | confidence={level1['confidence']:.2f}")
    if level1["blocking"]:
        print("Blocking:", ", ".join(level1["blocking"]))
    if "evidence_workspace" in report:
        ws = report["evidence_workspace"]
        print(
            f"Evidence: inventory={ws['inventory_count']} supported={ws['supported_count']} "
            f"unsupported={ws['unsupported_count']} extracted={ws['documents_extracted']}"
        )
        for item in ws["extraction_errors"]:
            print(f"Extraction error: {item['name']} -> {item['error']}")
    missing = [d for d in report["dependencies"] if not d["installed"]]
    if missing:
        print("Missing optional packages:", ", ".join(d["name"] for d in missing))
    else:
        print("Document/embedding dependencies: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
