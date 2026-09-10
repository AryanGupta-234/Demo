from __future__ import annotations

import json

from pre_cab.batch import run_batch
from pre_cab.input_loader import load_cr_records, normalize_cr_record


def _cr(number: str) -> dict[str, str]:
    return {
        "change_number": number,
        "change_type": "Normal",
        "short_description": "Routine patching",
        "implementation_plan": "Deploy patch and validate service health.",
        "rollback_plan": "Restore the previous version if validation fails.",
        "test_plan": "Check service health after deployment.",
        "risk_level": "Low",
        "cmdb_ci": "demo-service",
    }


def test_local_loader_accepts_nested_export_and_bom(tmp_path) -> None:
    source = tmp_path / "export.json"
    source.write_text(json.dumps({"result": [_cr("CHG-001")]}), encoding="utf-8-sig")

    records = load_cr_records(source)
    normalized = normalize_cr_record(records[0])

    assert normalized["Number"] == "CHG-001"
    assert normalized["Type"] == "Normal"
    assert normalized["Implementation plan"].startswith("Deploy")


def test_batch_normalizes_variants_and_checkpoints_duplicates(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    progress = run_batch([_cr("CHG-001"), _cr("CHG-001"), {"Type": "Emergency", "Number": "CHG-002"}], output_path=output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert progress.processed == 1
    assert progress.skipped == 2
    assert rows[0]["cr_number"] == "CHG-001"
    assert rows[0]["ok"] is True
    assert rows[0]["validation_mode"] == "evidence_aware"


def test_batch_can_screen_json_without_attachment_evidence(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    run_batch([_cr("CHG-003")], output_path=output, stage1_only=True)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["validation_mode"] == "metadata_only"
    assert row["stage2_decision"] is None
    assert row["final_decision"] == row["stage1_decision"]
