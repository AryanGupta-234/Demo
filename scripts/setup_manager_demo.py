#!/usr/bin/env python3
"""Prepare the complete local manager-demo workspace."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "demo"
WORKSPACE = ROOT / "data" / "manager_demo"
CR_SOURCE = SOURCE / "normal_crs.json"
SOURCE_ATTACHMENTS = SOURCE / "attachments"


def make_pdf(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setTitle(title)
    y = height - 60
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for raw_line in body.splitlines():
        line = raw_line[:110]
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
    c.save()


def main() -> int:
    records = json.loads(CR_SOURCE.read_text(encoding="utf-8"))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "crs").mkdir(exist_ok=True)
    (WORKSPACE / "evidence").mkdir(exist_ok=True)

    (WORKSPACE / "normal_crs.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    for cr in records:
        number = cr["Number"]
        cr_dir = WORKSPACE / "evidence" / number
        cr_dir.mkdir(parents=True, exist_ok=True)

        for source in SOURCE_ATTACHMENTS.glob(f"{number}_*"):
            shutil.copy2(source, cr_dir / source.name)

        make_pdf(
            cr_dir / f"{number}_change_summary.pdf",
            f"Change Evidence — {number}",
            "\n".join(
                [
                    f"Change: {number}",
                    f"Type: {cr.get('Type', '')}",
                    f"Category: {cr.get('Category', '')}",
                    f"Environment: {cr.get('Environment', '')}",
                    f"Risk: {cr.get('Risk', '')}",
                    "",
                    "Synthetic manager demonstration evidence.",
                    "This document is generated locally and contains no production data.",
                ]
            ),
        )

    manifest = {
        "purpose": "Pre-CAB Validator manager demonstration",
        "cr_count": len(records),
        "workspace": str(WORKSPACE),
        "evidence_root": str(WORKSPACE / "evidence"),
        "generated_files": sorted(
            str(p.relative_to(WORKSPACE))
            for p in WORKSPACE.rglob("*")
            if p.is_file()
        ),
    }
    (WORKSPACE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("MANAGER DEMO WORKSPACE READY")
    print("=" * 72)
    print(f"CR data      : {WORKSPACE / 'normal_crs.json'}")
    print(f"Evidence     : {WORKSPACE / 'evidence'}")
    print(f"CR count     : {len(records)}")
    print(f"Manifest     : {WORKSPACE / 'MANIFEST.json'}")
    print()
    print("Run demo:")
    print("  .venv\\Scripts\\python.exe scripts\\demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
