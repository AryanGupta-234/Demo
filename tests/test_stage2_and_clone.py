from pre_cab.clone import change_similarity, clone_analysis
from pre_cab.evidence import EvidenceDocument, verify_attachments
from pre_cab.schemas import Decision, Requirement, Strictness


def test_missing_customer_approval_document_blocks_claim():
    cr = {
        "Number": "CHG-DEMO-010",
        "Customer Approval": "Yes",
        "Test plan": "UAT completed",
        "Test Results Evidence": "Yes",
        "Backout plan": "Restore the previous application artifact from backup.",
    }
    docs = [EvidenceDocument("test-1", "uat.txt", "CHG-DEMO-010\nUAT test cases\nExpected result: pass\nActual result: pass")]
    result = verify_attachments(
        cr,
        docs,
        requirements=[Requirement("UAT", True, "functional change")],
        strictness=Strictness.BALANCED,
    )
    assert result.decision == Decision.NOT_READY
    assert result.verified["testing"] is True
    assert result.verified["customer_approval"] is False


def test_uat_dev_only_contradiction_is_blocking():
    cr = {
        "Number": "CHG-DEMO-011",
        "Test plan": "Successfully tested in UAT",
        "Test Results Evidence": "Yes",
        "Backout plan": "Restore the previous release from backup.",
    }
    docs = [EvidenceDocument("test-2", "test-report.txt", "CHG-DEMO-011\nDEV testing completed\nExpected result: pass")]
    result = verify_attachments(cr, docs, requirements=[Requirement("UAT", True, "customer-facing change")])
    assert result.decision == Decision.NOT_READY
    assert any("environment" in text.lower() for text in result.contradictions)


def test_strong_clone_candidate_and_delta():
    historical = {
        "Number": "CHG-HIST-001",
        "Type": "Normal",
        "Category": "Core",
        "Sub Category": "Branch",
        "Short description": "Deploy application enhancement",
        "Description": "Customer-facing enhancement to transaction behavior.",
        "Implementation plan": "Backup artifact and deploy new application version, then restart service.",
        "Test plan": "UAT functional validation completed.",
        "Backout plan": "Restore previous application artifact from backup.",
        "Customer Approval": "Yes",
        "Risk": "Moderate",
        "Configuration item": "demo-prod",
        "CAB Outcome": "Approved",
    }
    current = dict(historical)
    current["Number"] = "CHG-NEW-001"
    current["Short description"] = "Deploy application enhancement - September release"
    current["Customer Approval"] = "Requested"
    score = change_similarity(current, historical)
    analysis = clone_analysis(current, historical, score)
    assert score >= 0.90
    assert analysis.recommendation == "CLONE_CANDIDATE"
    assert "Customer Approval" in analysis.changed_fields
