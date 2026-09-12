from pre_cab.agents import FieldAgent
from pre_cab.schemas import AgentContext, Strictness


def test_field_agent_selects_decision_relevant_fields_and_excludes_noise():
    cr = {
        "Number": "CHG001",
        "Type": "Normal",
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Short description": "Deploy application release",
        "Description": "Deploy release to production after lower-environment testing.",
        "Justification": "Required business release.",
        "Implementation plan": "Deploy package, validate service health, and confirm smoke tests.",
        "Change plan": "legacy unused field should not matter",
        "Backout plan": "Restore previous package and restart service if validation fails.",
        "Test plan": "Smoke and regression tests in lower environment.",
        "Configuration item": "prod-app-01",
        "Risk": "3",
        "Priority": "3 - Moderate",
        "UAT signoff": "Not Applicable",
        "Customer Approval": "Yes",
        "TCS QA signoff": "Not Applicable",
        "Test Results Evidence": "Yes",
        "Lower Environment Reference CR/SR": "CHG000999",
        "Comments and Work notes": "large noisy notes that do not need to go to the LLM",
        "Random operational field": "irrelevant payload",
    }
    result = FieldAgent().run(AgentContext(cr=cr, strictness=Strictness.BALANCED))

    selected = result.notes["selected_fields"]
    selected_cr = result.notes["selected_cr"]

    assert "Implementation plan" in selected
    assert "Backout plan" in selected
    assert "Test plan" in selected
    assert "Customer Approval" in selected
    assert "Lower Environment Reference CR/SR" in selected
    assert "Change plan" not in selected
    assert "Comments and Work notes" not in selected
    assert "Random operational field" not in selected
    assert set(selected_cr) == set(selected)
    assert result.notes["selected_field_count"] < result.notes["field_count"]


def test_field_agent_tracks_not_observed_fields():
    cr = {
        "Number": "CHG001",
        "Type": "Normal",
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Implementation plan": "Deploy package.",
        "Change plan": "",
    }
    result = FieldAgent().run(AgentContext(cr=cr, strictness=Strictness.BALANCED))
    assert "Change plan" in result.notes["not_observed_omitted"]
