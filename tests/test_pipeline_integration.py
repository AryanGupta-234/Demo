from __future__ import annotations

from pathlib import Path

from pre_cab.pipeline import _merge_decisions, run_pre_cab
from pre_cab.schemas import Decision, Strictness


def _normal_cr(number: str = "CHG-DEMO-100") -> dict:
    return {
        "Number": number,
        "Type": "Normal",
        "Short description": "Application enhancement for customer transaction",
        "Description": "Deploy updated application component for customer transaction flow",
        "Justification": "Fix functional defect",
        "Implementation plan": "Back up current component, deploy new component, restart service",
        "Backout plan": "Restore backup and restart service",
        "Test plan": "UAT completed with functional scenarios",
        "UAT signoff": "Yes",
        "Customer Approval": "Yes",
        "Conflict status": "No Conflict",
        "Configuration item": "Demo Production Application",
    }


def test_merge_decisions_never_upgrades_a_weaker_stage() -> None:
    assert _merge_decisions(Decision.PASS, Decision.PASS) == Decision.PASS
    assert _merge_decisions(Decision.CONDITIONAL, Decision.PASS) == Decision.CONDITIONAL
    assert _merge_decisions(Decision.PASS, Decision.CONDITIONAL) == Decision.CONDITIONAL
    assert _merge_decisions(Decision.PASS, Decision.NOT_READY) == Decision.NOT_READY
    assert _merge_decisions(Decision.NOT_READY, Decision.PASS) == Decision.NOT_READY


def test_pipeline_discovers_local_attachment(tmp_path: Path) -> None:
    cr = _normal_cr()
    evidence = tmp_path / "CHG-DEMO-100_UAT.txt"
    evidence.write_text(
        "CHG-DEMO-100 UAT test case expected result actual result PASS customer approval approved by customer",
        encoding="utf-8",
    )

    result = run_pre_cab(cr, attachment_root=tmp_path, strictness=Strictness.BALANCED)

    assert len(result.documents) == 1
    assert result.stage2 is not None
    assert result.stage2.verified["testing"] is True
    assert result.stage2.verified["customer_approval"] is True


def test_pipeline_does_not_guess_unrelated_attachments(tmp_path: Path) -> None:
    cr = _normal_cr()
    (tmp_path / "OTHER-CR_UAT.txt").write_text("test case expected result PASS", encoding="utf-8")

    result = run_pre_cab(cr, attachment_root=tmp_path, strictness=Strictness.BALANCED)

    assert result.documents == []


def test_pipeline_accepts_generator_evidence_files_and_reports_inventory(tmp_path: Path) -> None:
    cr = _normal_cr("CHG-DEMO-GEN")
    source = tmp_path / "CHG-DEMO-GEN_UAT.txt"
    source.write_text("CHG-DEMO-GEN UAT test cases expected result actual result PASS", encoding="utf-8")
    unsupported = tmp_path / "CHG-DEMO-GEN_notes.bin"
    unsupported.write_bytes(b"binary")

    result = run_pre_cab(
        cr,
        attachment_root=tmp_path / "workspace",
        evidence_files=(item for item in [source]),
        strictness=Strictness.BALANCED,
    )

    assert len(result.documents) == 1
    assert result.stage1.metadata["evidence_workspace"]["files_supplied"] == 1
    inventory = result.evidence_manifest["workspace_inventory"]
    assert any(item["name"] == "CHG-DEMO-GEN_UAT.txt" for item in inventory)
