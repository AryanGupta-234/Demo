from pre_cab.narrative import _checklist, _recommendations, format_cab_result
from pre_cab.schemas import Decision, Finding, FindingSeverity


def _finding(code, title, severity, recommendation=""):
    return Finding(code, title, severity, title, technical_detail="", recommendation=recommendation)


def test_checklist_surfaces_unrepresented_warnings():
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
    assert any(line.startswith("⚠️ Missing Target environment") for line in lines)


def test_checklist_does_not_duplicate_already_represented_findings():
    findings = [
        _finding("MISSING_CUSTOMER_APPROVAL", "Missing Customer approval", FindingSeverity.WARNING),
        _finding("BACKOUT_OK", "Recovery path identified", FindingSeverity.INFO),
    ]
    cr = {"Category": "Batch", "Customer Approval": ""}
    lines = _checklist(cr, [], None, findings)
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


def test_recommendations_fall_back_to_deterministic_only_without_brain_payload():
    findings = [_finding("X", "x", FindingSeverity.WARNING, recommendation="Do the thing.")]
    recs = _recommendations({}, findings, None)
    assert recs == ["Do the thing."]


def test_format_cab_result_end_to_end_is_deep_and_evidence_focused():
    findings = [
        _finding("MISSING_ENVIRONMENT", "Missing Target environment", FindingSeverity.WARNING),
    ]
    cr = {
        "Category": "Batch",
        "Short description": "Restart application service",
        "Description": "Restart production service after configuration change.",
        "Justification": "Required maintenance activity.",
        "Implementation plan": "Stop service; apply change; start service; verify health.",
        "Change plan": "Sequence with dependent service restart.",
        "Backout plan": "Restore prior config and restart service.",
        "Test plan": "Validate health endpoint and key transaction after restart.",
        "Configuration item": "CI-123",
        "Risk": "Medium",
        "Priority": "2",
        "Test Results Evidence": "No attachment supplied",
        "Lower Environment Reference CR/SR": "CR-123",
        "Customer Approval": "Yes",
        "UAT signoff": "Not Applicable",
        "TCS QA signoff": "Yes",
    }
    text = format_cab_result("CHG-TEST", cr, Decision.CONDITIONAL, 0.9, [], findings, {})
    assert "EXECUTIVE SUMMARY" in text
    assert "EVIDENCE REVIEWED" in text
    assert "TECHNICAL ASSESSMENT" in text
    assert "TESTING & VALIDATION ASSESSMENT" in text
    assert "RISK & OPERATIONAL IMPACT" in text
    assert "DECISION TRACE" in text
    assert "CAB QUESTIONS" in text
    assert "REQUIRED ACTIONS BEFORE CAB" in text
    assert "Implementation procedure is populated" in text
    assert "Missing Target environment" in text
    assert "CHG-TEST" in text


def test_model_summary_uses_the_real_output_contract_keys():
    """Regression test: format_cab_result's Executive Summary must be built
    from the keys the model is actually asked for (cab_reasoning /
    technical_reasoning per brain_loop.py's output_contract), not a
    "executive_summary" key that was never part of that contract and so
    never populated in a real run — see _decision_summary in narrative.py."""
    cr = {"Category": "Batch", "Implementation plan": "steps"}
    brain = {
        "cab_reasoning": "The change has a material testing gap that CAB should not approve yet.",
        "technical_reasoning": "Test plan is populated but no execution evidence field is filled.",
    }
    text = format_cab_result("CHG-TEST", cr, Decision.NOT_READY, 0.75, [], [], brain)
    assert "material testing gap" in text
    assert "Technical basis:" in text
    assert "no execution evidence field is filled" in text


def test_model_summary_falls_back_to_template_without_cab_reasoning():
    """If the model returns a payload with no cab_reasoning (e.g. an older/
    malformed response), the Executive Summary must still be the
    deterministic template — never blank, and never the raw
    technical_reasoning text standing in for it."""
    cr = {"Category": "Batch"}
    brain = {"technical_reasoning": "only technical detail, no cab_reasoning"}
    text = format_cab_result("CHG-TEST", cr, Decision.NOT_READY, 0.75, [], [], brain)
    summary_section = text.split("EXECUTIVE SUMMARY")[1].split("Decision drivers:")[0]
    assert "not CAB-ready" in summary_section
    assert "only technical detail, no cab_reasoning" not in summary_section
