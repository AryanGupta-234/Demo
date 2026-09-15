from pre_cab.narrative import _checklist, _recommendations, format_cab_result
from pre_cab.schemas import Decision, Finding, FindingSeverity


def _finding(code, title, severity, recommendation=""):
    return Finding(code, title, severity, title, technical_detail="", recommendation=recommendation)


def test_checklist_surfaces_unrepresented_warnings():
    # Regression: a real report (CHG0063542) showed all 7 curated checklist
    # items as green while CONDITIONAL was driven entirely by 3 WARNING
    # findings (Missing Target environment, Change conflict flagged, Field
    # quality weak) that never appeared anywhere in the checklist - looked
    # untrustworthy ("why CONDITIONAL if everything is checked?").
    findings = [
        _finding("MISSING_ENVIRONMENT", "Missing Target environment", FindingSeverity.WARNING),
        _finding("CONFLICT", "Change conflict flagged", FindingSeverity.WARNING),
        _finding("FIELD_QUALITY_WEAK", "One or more populated fields are weak or placeholder-like", FindingSeverity.WARNING),
    ]
    cr = {
        "Category": "Batch", "Configuration item": "X", "Implementation plan": "steps",
        "Test plan": "plan", "Customer Approval": "Yes",
    }
    lines = _checklist(cr, [], None, findings)
    assert any("Missing Target environment" in line for line in lines)
    assert any("Change conflict flagged" in line for line in lines)
    assert any("weak or placeholder-like" in line for line in lines)
    # WARNING findings get a warning mark, not a hard cross
    assert any(line.startswith("⚠️ Missing Target environment") for line in lines)


def test_checklist_does_not_duplicate_already_represented_findings():
    findings = [
        _finding("MISSING_CUSTOMER_APPROVAL", "Missing Customer approval", FindingSeverity.WARNING),
        _finding("BACKOUT_OK", "Recovery path identified", FindingSeverity.INFO),
    ]
    cr = {"Category": "Batch", "Customer Approval": ""}
    lines = _checklist(cr, [], None, findings)
    # already covered by the curated "Customer approval = X" / rollback lines -
    # should not also appear as a second, separate bullet
    assert not any("Missing Customer approval" in line for line in lines)
    assert not any("Recovery path identified" in line for line in lines)


def test_recommendations_combine_deterministic_and_model_without_duplication():
    findings = [
        _finding("MISSING_ENVIRONMENT", "x", FindingSeverity.WARNING, recommendation="Populate target environment before CAB review."),
    ]
    brain = {"recommendations": ["Populate target environment before CAB review.", "Monitor the rollout closely."]}
    recs = _recommendations({}, findings, brain)
    assert recs.count("Populate target environment before CAB review.") == 1
    assert "Monitor the rollout closely." in recs
    assert recs[-1] == "Confirm final CAB evidence before approval."


def test_recommendations_fall_back_to_deterministic_only_without_brain_payload():
    findings = [_finding("X", "x", FindingSeverity.WARNING, recommendation="Do the thing.")]
    recs = _recommendations({}, findings, None)
    assert recs == ["Do the thing.", "Confirm final CAB evidence before approval."]


def test_format_cab_result_end_to_end_shows_all_decision_drivers():
    findings = [
        _finding("MISSING_ENVIRONMENT", "Missing Target environment", FindingSeverity.WARNING),
    ]
    cr = {
        "Category": "Batch", "Configuration item": "X", "Implementation plan": "steps",
        "Test plan": "plan", "Customer Approval": "Yes",
    }
    text = format_cab_result("CHG-TEST", cr, Decision.CONDITIONAL, 0.9, [], findings, {})
    assert "Missing Target environment" in text
    assert "CONDITIONAL" in text
