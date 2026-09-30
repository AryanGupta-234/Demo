from __future__ import annotations

import json
from pathlib import Path

from pre_cab import ocr
from pre_cab.local_attachments import load_attachments_for_cr
from pre_cab.rule_lifecycle import automatically_promote


def test_image_ocr_execution_is_used_for_evidence(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "evidence"
    folder = root / "CHG-OCR-001"
    folder.mkdir(parents=True)
    image = folder / "CHG-OCR-001_screenshot.png"
    image.write_bytes(b"synthetic image bytes")

    monkeypatch.delenv("PRE_CAB_VISION_MODEL", raising=False)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda path: (
            "CHG-OCR-001\nUAT completed\nActual Result: PASS",
            {
                "ocr_engine": "fake-tesseract",
                "ocr_executed": True,
                "ocr_confidence": 0.93,
                "ocr_text_chars": 43,
            },
        ),
    )

    docs = load_attachments_for_cr(root, "CHG-OCR-001")
    assert len(docs) == 1
    assert docs[0].metadata["ocr_executed"] is True
    assert docs[0].metadata["requires_vision"] is True
    assert "UAT completed" in docs[0].text


def test_automatic_rule_promotion_reaches_active_on_stable_holdout() -> None:
    candidate = {
        "rule_table_version": 1,
        "lifecycle_status": "CANDIDATE",
        "buckets": {},
        "categories": {},
        "global": {
            "support": 100,
            "fields": {
                "Implementation plan": {
                    "kind": "descriptive",
                    "requirement_level": "REQUIRED",
                    "support": 100,
                    "fill_rate": 1.0,
                }
            },
        },
    }
    holdout = [
        {
            "Type": "Normal",
            "Category": "Infrastructure",
            "Sub Category": "Patching",
            "Implementation plan": "Back up, deploy, validate.",
        }
        for _ in range(100)
    ]

    promoted = automatically_promote(candidate, holdout)
    assert promoted["lifecycle_status"] == "ACTIVE"
    assert promoted["validation"]["stability_rate"] == 1.0
    assert promoted["lifecycle_history"][-1]["to"] == "ACTIVE"


def test_rule_promotion_stays_candidate_with_insufficient_holdout() -> None:
    candidate = {
        "rule_table_version": 1,
        "lifecycle_status": "CANDIDATE",
        "buckets": {},
        "categories": {},
        "global": {
            "fields": {
                "Implementation plan": {
                    "kind": "descriptive",
                    "requirement_level": "REQUIRED",
                }
            }
        },
    }
    holdout = [{"Type": "Normal", "Implementation plan": "Deploy."} for _ in range(20)]
    promoted = automatically_promote(candidate, holdout, min_holdout_records=100)
    assert promoted["lifecycle_status"] == "CANDIDATE"
    assert promoted["validation"]["normal_holdout_record_count"] == 20
