from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pre_cab.evidence import EvidenceDocument
from pre_cab.local_attachments import discover_attachments, inventory_workspace
from pre_cab.parallel import process_crs_parallel
from pre_cab.pipeline import _merge_decisions, run_pre_cab
from pre_cab.schemas import Decision, Strictness


def _cr(number: str) -> dict:
    return {
        "Number": number,
        "Type": "Normal",
        "Short description": f"Application change for {number}",
        "Description": f"Implement and validate the planned application change for {number}.",
        "Justification": "Required operational change.",
        "Implementation plan": "Back up current artifact, deploy the approved change, run smoke validation.",
        "Backout plan": "Restore the previous artifact and validate service health.",
        "Test plan": "Run smoke tests and record execution results.",
        "Customer Approval": "Yes",
        "UAT signoff": "Not Applicable",
        "Conflict status": "No Conflict",
        "Configuration item": "Demo Application",
        "Risk": "Moderate",
    }


def test_parallel_processing_preserves_input_order_and_cr_identity(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    for number in ("CHG-PAR-001", "CHG-PAR-002"):
        folder = root / number
        folder.mkdir(parents=True)
        (folder / f"{number}_test.txt").write_text(
            f"{number} test execution result PASS",
            encoding="utf-8",
        )

    results = process_crs_parallel(
        [_cr("CHG-PAR-001"), _cr("CHG-PAR-002")],
        attachment_root=str(root),
        max_workers=2,
    )

    assert [item.cr_number for item in results] == ["CHG-PAR-001", "CHG-PAR-002"]
    assert all(item.error is None for item in results)
    for item in results:
        assert item.result is not None
        refs = [document.ref for document in item.result.documents]
        assert refs
        assert all(item.cr_number in ref for ref in refs)
        other = "CHG-PAR-002" if item.cr_number == "CHG-PAR-001" else "CHG-PAR-001"
        assert all(other not in ref for ref in refs)


def test_workspace_inventory_keeps_unsupported_files_visible(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    folder = root / "CHG-INV-001"
    folder.mkdir(parents=True)
    (folder / "supported.txt").write_text("evidence", encoding="utf-8")
    (folder / "unsupported.bin").write_bytes(b"binary")

    inventory = inventory_workspace(root, "CHG-INV-001")

    statuses = {item["name"]: item["status"] for item in inventory}
    assert statuses["supported.txt"] == "supported"
    assert statuses["unsupported.bin"] == "unsupported"


def test_symlinked_cr_workspace_is_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "CHG-SYM-001_test.txt").write_text("PASS", encoding="utf-8")

    link = root / "CHG-SYM-001"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform")

    assert discover_attachments(root, "CHG-SYM-001") == []


def test_final_gate_remains_monotonic() -> None:
    assert _merge_decisions(Decision.PASS, Decision.PASS) == Decision.PASS
    assert _merge_decisions(Decision.PASS, Decision.CONDITIONAL) == Decision.CONDITIONAL
    assert _merge_decisions(Decision.CONDITIONAL, Decision.PASS) == Decision.CONDITIONAL
    assert _merge_decisions(Decision.PASS, Decision.NOT_READY) == Decision.NOT_READY
    assert _merge_decisions(Decision.NOT_READY, Decision.PASS) == Decision.NOT_READY


def test_pipeline_records_cr_scoped_evidence_manifest(tmp_path: Path) -> None:
    number = "CHG-MANIFEST-001"
    root = tmp_path / "evidence"
    folder = root / number
    folder.mkdir(parents=True)
    (folder / f"{number}_evidence.txt").write_text(
        f"{number} test execution result PASS",
        encoding="utf-8",
    )

    result = run_pre_cab(_cr(number), attachment_root=root, strictness=Strictness.BALANCED)

    assert result.evidence_manifest["cr_number"] == number
    assert result.evidence_manifest["inventory_count"] >= 1
    assert result.evidence_manifest["document_count"] == 1
    assert result.stage1.metadata["evidence_workspace"]["cr_folder"].endswith(number)
