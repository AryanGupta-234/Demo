from pre_cab.decision import validate_fields
from pre_cab.requirements import infer_requirements
from pre_cab.rules import context_flags, field_policies, required_field_policies, contextual_signals
from pre_cab.schemas import Decision, Strictness


def base_cr(**overrides):
    data = {
        "Number": "CHG-TEST-001",
        "Type": "Normal",
        "Short description": "Deploy transaction service enhancement",
        "Description": "Customer-facing transaction enhancement for production users.",
        "Justification": "Improve transaction processing and correct a defect.",
        "Implementation plan": "Backup current artifact, deploy release, restart service, validate smoke tests.",
        "Backout plan": "Restore the previous release from backup and restart the service.",
        "Test plan": "Execute functional scenarios and production smoke validation.",
        "Risk": "Moderate",
        "Risk and impact analysis": "Customer-facing transaction change with controlled maintenance impact.",
        "Configuration item": "wallet-prod",
        "Environment": "Production",
        "Category": "Application",
        "Sub Category": "Transactions",
        "Conflict status": "No Conflict",
    }
    data.update(overrides)
    return data


def test_rule_catalog_has_baseline_and_conditional_fields():
    policies = field_policies()
    assert len(policies) >= 20
    assert {p.field for p in policies if p.baseline} >= {
        "Number", "Type", "Short description", "Description", "Justification",
        "Implementation plan", "Backout plan", "Test plan", "Risk",
    }
    assert any(p.field == "Customer Approval" and p.required_when for p in policies)
    assert any(p.field == "UAT signoff" and p.required_when for p in policies)


def test_customer_facing_change_requires_customer_approval_and_uat():
    flags = context_flags(base_cr())
    assert flags["customer-impact"] is True
    assert flags["uat"] is True
    required = {policy.field for policy, enabled, _ in required_field_policies(base_cr()) if enabled}
    assert "Customer Approval" in required
    assert "UAT signoff" in required


def test_infrastructure_patch_does_not_force_uat():
    cr = base_cr(
        **{
            "Short description": "Monthly OS security patch",
            "Description": "Apply monthly OS patching and reboot servers.",
            "Category": "Infrastructure",
            "Sub Category": "OS",
        }
    )
    flags = context_flags(cr)
    assert flags["infrastructure"] is True
    assert flags["uat"] is False
    uat = next(signal for signal in contextual_signals(cr) if signal.requirement == "UAT")
    assert uat.applicable is False


def test_missing_contextual_field_is_evaluated_by_decision_gate():
    cr = base_cr(**{"Customer Approval": ""})
    result = validate_fields(cr, Strictness.BALANCED)
    assert any(f.code == "MISSING_CUSTOMER_APPROVAL" for f in result.findings)
    assert result.decision == Decision.CONDITIONAL


def test_strict_mode_turns_contextual_gap_into_not_ready():
    cr = base_cr(**{"Customer Approval": ""})
    result = validate_fields(cr, Strictness.STRICT)
    assert result.decision == Decision.NOT_READY


def test_rule_engine_exposes_explainable_context_to_requirements():
    predictions = infer_requirements(base_cr())
    names = {prediction.name for prediction in predictions}
    assert "Implementation plan" in names
    assert "UAT signoff" in names
    assert any(prediction.name == "UAT" and prediction.required for prediction in predictions)


def test_prod_workflow_derives_unset_environment_without_missing_target_warning():
    cr = base_cr(
        **{
            "Environment": "",
            "Category": "Infrastructure",
            "Short description": "Monthly Windows OS security patch",
            "Description": "Apply monthly security patches and reboot production management servers.",
        }
    )
    result = validate_fields(cr, Strictness.BALANCED)
    assert not any(f.code == "MISSING_ENVIRONMENT" for f in result.findings)
    assert any(f.code == "TARGET_ENVIRONMENT_DERIVED" for f in result.findings)


def test_missing_ci_is_traceability_observation_not_technical_blocker():
    cr = base_cr(**{"Configuration item": "", "Category": "Infrastructure"})
    result = validate_fields(cr, Strictness.BALANCED)
    finding = next(f for f in result.findings if f.code == "CONFIGURATION_TRACEABILITY")
    assert finding.severity == FindingSeverity.INFO
    assert not any(f.code == "MISSING_CONFIGURATION_ITEM" for f in result.findings)


def test_non_prod_validation_treats_uat_sit_and_preprod_as_equivalent():
    base = base_cr(**{
        "Category": "Infrastructure",
        "Short description": "Server maintenance",
        "Description": "Infrastructure maintenance change.",
    })
    for phrase in ("tested in UAT", "tested in SIT", "tested in Pre-PROD", "validated in lower environment"):
        cr = dict(base)
        cr["Test plan"] = phrase
        flags = context_flags(cr)
        assert flags["non-prod-validation"] is True


def test_non_prod_environment_mention_without_testing_is_not_validation_claim():
    cr = base_cr(**{
        "Category": "Infrastructure",
        "Justification": "There are no Windows servers present in Non-Prod environment.",
        "Test plan": "Validate production server health after patching.",
    })
    assert context_flags(cr)["non-prod-validation"] is False


def test_model_facing_field_sets_include_exact_required_14():
    from pre_cab.rules import MODEL_DESCRIPTIVE_FIELDS, MODEL_SIGNOFF_FIELDS
    assert list(MODEL_DESCRIPTIVE_FIELDS) == [
        "Short description", "Description", "Justification", "Implementation plan",
        "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
    ]
    assert list(MODEL_SIGNOFF_FIELDS) == [
        "UAT signoff", "Customer Approval", "TCS QA signoff",
        "Test Results Evidence", "Lower Environment Reference CR/SR",
    ]
