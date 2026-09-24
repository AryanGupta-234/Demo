from pre_cab.decision import validate_fields
from pre_cab.requirements import infer_requirements
from pre_cab.rules import context_flags, field_policies, required_field_policies, contextual_signals
from pre_cab.schemas import FindingSeverity
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


def test_infrastructure_change_does_not_require_formal_test_results_evidence():
    cr = base_cr(
        **{
            "Short description": "Microsoft Edge security update on Windows servers",
            "Description": "Apply the Microsoft Edge security update to production Windows servers. No application outage is expected.",
            "Justification": "Remediate a security vulnerability in the server tooling.",
            "Risk and impact analysis": "No application or customer-facing service change is expected; this is infrastructure maintenance.",
            "Category": "Infrastructure",
            "Sub Category": "",
            "Risk": "",
            "Planned start": "",
            "Planned end": "",
            "Conflict status": "",
            "Test plan": "Verify the installed Edge version and perform a post-change sanity check.",
            "Test Results Evidence": "",
        }
    )
    flags = context_flags(cr)
    assert flags["infrastructure"] is True
    assert flags["functional"] is False
    assert flags["testing-evidence"] is False
    required = {policy.field for policy, enabled, _ in required_field_policies(cr) if enabled}
    assert "Test Results Evidence" not in required

    result = validate_fields(cr, Strictness.BALANCED)
    assert not any(f.code == "MISSING_TEST_RESULTS_EVIDENCE" for f in result.findings)
    material = [(f.code, f.severity.value, f.title) for f in result.findings if f.severity != FindingSeverity.INFO]
    assert result.decision == Decision.PASS, material


def test_governance_metadata_gaps_are_visible_but_do_not_change_readiness():
    cr = base_cr(
        **{
            "Risk": "",
            "Sub Category": "",
            "Planned start": "",
            "Planned end": "",
            "Conflict status": "",
            "Category": "Infrastructure",
            "Short description": "Windows server maintenance",
            "Description": "Routine production infrastructure maintenance with no customer-facing application change.",
            "Justification": "Maintain production infrastructure; application behavior is unchanged.",
            "Risk and impact analysis": "No customer-facing service impact is expected from this infrastructure maintenance.",
            "Test plan": "Perform post-change health and service checks.",
        }
    )
    result = validate_fields(cr, Strictness.BALANCED)
    advisory_codes = {
        "MISSING_RISK", "MISSING_SUB_CATEGORY", "MISSING_PLANNED_START",
        "MISSING_PLANNED_END", "CONFLICT_UNVERIFIED",
    }
    present = {f.code: f.severity for f in result.findings if f.code in advisory_codes}
    assert all(severity == FindingSeverity.INFO for severity in present.values())
    material = [(f.code, f.severity.value, f.title) for f in result.findings if f.severity != FindingSeverity.INFO]
    assert result.decision == Decision.PASS, material


def test_generic_security_infrastructure_language_does_not_become_functional():
    cr = base_cr(
        **{
            "Category": "Infrastructure",
            "Sub Category": "OS Patching",
            "Short description": "Security update for production servers",
            "Description": "Apply a critical security patch to Windows servers. The application remains unchanged.",
            "Justification": "Remediate a vulnerability.",
            "Test plan": "Verify service health and installed patch version after maintenance.",
            "Risk": "",
            "Test Results Evidence": "",
        }
    )
    from pre_cab.rules import change_profile
    profile = change_profile(cr)
    assert profile["infrastructure"] is True
    assert profile["security"] is True
    assert profile["functional"] is False
    assert profile["formal_test_evidence_expected"] is False


def test_non_prod_validation_is_not_dependency_evidence():
    cr = base_cr(
        **{
            "Category": "Infrastructure",
            "Sub Category": "OS Patching",
            "Change plan": "",
            "Test plan": "Tested in SIT before the production patch.",
            "Description": "Apply the server patch in production.",
            "Short description": "Production server patch",
        }
    )
    from pre_cab.agents import TechnicalAgent
    from pre_cab.schemas import AgentContext
    result = TechnicalAgent().run(AgentContext(cr=cr))
    assert result.notes["non_prod_validation_claimed"] is True
    assert result.notes["dependency_evidence_present"] is False


def test_historical_field_layer_does_not_reintroduce_metadata_warnings():
    cr = base_cr(
        **{
            "Risk": "",
            "Sub Category": "",
            "Planned start": "",
            "Planned end": "",
            "Conflict status": "",
            "Category": "Infrastructure",
            "Short description": "Production server maintenance",
            "Description": "Routine infrastructure maintenance.",
            "Test plan": "Verify server health after maintenance.",
        }
    )
    result = validate_fields(cr, Strictness.BALANCED)
    forbidden = {
        "HISTORICAL_GAP_RISK", "HISTORICAL_GAP_SUB_CATEGORY",
        "HISTORICAL_GAP_PLANNED_START", "HISTORICAL_GAP_PLANNED_END",
        "HISTORICAL_GAP_CONFLICT_STATUS",
    }
    assert not any(f.code in forbidden for f in result.findings)
