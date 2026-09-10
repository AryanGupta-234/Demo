from pre_cab.decision import infer_uat_requirement, rollback_quality, validate_fields
from pre_cab.schemas import Decision, Strictness


def base_cr(**overrides):
    data = {
        "Number": "CHG-DEMO-001",
        "Type": "Normal",
        "Short description": "Deploy application enhancement",
        "Description": "Customer-facing enhancement to transaction behavior.",
        "Justification": "Fix and improve transaction processing.",
        "Implementation plan": "Backup artifact, deploy new version, restart service, validate.",
        "Backout plan": "Take backup of the existing project and restore it if deployment fails.",
        "Test plan": "Validated functional scenarios in UAT.",
        "UAT signoff": "Not Applicable",
        "Test Results Evidence": "Yes",
        "Risk": "Moderate",
        "Risk and impact analysis": "Low customer impact during the maintenance window.",
        "Configuration item": "demo-prod",
        "Conflict status": "No Conflict",
    }
    data.update(overrides)
    return data


def test_backout_backup_is_accepted():
    ok, _ = rollback_quality("take back up of existing wallet project")
    assert ok is True


def test_uat_is_contextual_for_infrastructure_patch():
    cr = base_cr(
        Category="Infrastructure",
        Short_description="" if False else "Monthly OS security patching",
        Description="Monthly OS security patching and server reboot.",
        Justification="Remediate operating system vulnerabilities.",
    )
    required, _ = infer_uat_requirement(cr)
    assert required is False


def test_functional_change_can_require_uat():
    cr = base_cr(Category="Debit Card", Sub_Category="ATM Transactions")
    required, _ = infer_uat_requirement(cr)
    assert required is True


def test_conflict_blocks_balanced_stage_one():
    result = validate_fields(base_cr(**{"Conflict status": "Conflict"}), Strictness.BALANCED)
    assert result.decision == Decision.NOT_READY


def test_missing_implementation_blocks_balanced_stage_one():
    result = validate_fields(base_cr(**{"Implementation plan": ""}), Strictness.BALANCED)
    assert result.decision == Decision.NOT_READY
