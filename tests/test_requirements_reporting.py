from pre_cab.requirements import infer_requirements
from pre_cab.reporting import build_report
from pre_cab.schemas import Decision, Finding, FindingSeverity, Strictness, ValidationResult


def test_infrastructure_patch_does_not_force_uat():
    cr = {
        "Type": "Normal",
        "Short description": "Monthly OS patching",
        "Description": "Apply security patches and reboot production servers.",
        "Justification": "Security remediation.",
        "Category": "Infrastructure",
    }
    results = {r.name: r for r in infer_requirements(cr)}
    assert results["UAT"].required is False
    assert results["Rollback / recovery"].required is True


def test_customer_facing_change_predicts_customer_approval():
    cr = {
        "Type": "Normal",
        "Short description": "Customer SMS enhancement",
        "Description": "Change SMS content for customer payments.",
        "Justification": "Improve customer notifications.",
        "Category": "Core",
        "Sub Category": "branchchannel",
    }
    results = {r.name: r for r in infer_requirements(cr)}
    assert results["UAT"].required is True
    assert results["Customer Approval"].required is True


def test_report_contains_both_views():
    validation = ValidationResult(
        decision=Decision.PASS,
        confidence=0.94,
        score=94,
        strictness=Strictness.BALANCED,
        findings=[Finding("OK", "Ready", FindingSeverity.INFO, "Looks ready", technical_detail="Detailed technical evidence")],
        cab_summary="Ready for CAB",
        technical_summary="Full technical analysis",
    )
    report = build_report(validation)
    assert report["decision"] == "PASS"
    assert report["cab_view"]["summary"] == "Ready for CAB"
    assert report["technical_view"]["informational"][0]["technical_detail"] == "Detailed technical evidence"
