            lines.append(f"  Risk assessment confidence: {float(notes['risk_confidence']):.0%}")
        lines.append("")
    return lines[:-1] if lines else []


def _cab_questions(cr: dict[str, Any], agent_results: list[Any], brain_payload: dict[str, Any] | None) -> list[str]:
    if brain_payload and isinstance(brain_payload.get("cab_questions"), list) and brain_payload["cab_questions"]:
        return [str(q) for q in brain_payload["cab_questions"][:6]]

    by_name = {result.agent: result for result in agent_results}
    technical = by_name.get("technical")
    testing = by_name.get("testing")
    risk = by_name.get("risk")
    questions: list[str] = []
    context_result = by_name.get("context")
    profile = (context_result.notes.get("change_profile") if context_result else None) or {}
    # Traceability is useful context but should only become a CAB question when it
    # materially affects understanding of the target scope.
    if not _text(cr.get("Configuration item")) and profile.get("primary_archetype") in {"FUNCTIONAL", "DATABASE", "NETWORK"}:
        questions.append("Which production Configuration Item(s) are actually being changed?")
    if testing and testing.notes.get("functional_coverage_ok") is False:
        questions.append("What validation is still needed to demonstrate the intended outcome?")
    elif testing and profile.get("formal_test_evidence_expected") and not testing.notes.get("execution_claimed") and not testing.notes.get("formal_evidence_present"):
        questions.append("When the planned validation is executed, where will the result be recorded?")
    if risk and not _text(cr.get("Risk")) and (profile.get("high_impact") or profile.get("customer_facing") or profile.get("security")):
        questions.append("What is the formal risk classification and who assessed it?")
    if technical and not technical.notes.get("rollback_aligned"):
        questions.append("What is the exact rollback trigger, procedure, and recovery validation?")
    if technical and technical.notes.get("dependency_evidence_present") is False and profile.get("high_impact"):
        questions.append("Is there any dependency or sequencing detail that the implementation team must coordinate?")
    return questions[:6] or ["Are there any residual operational risks or dependencies not captured in the CR?"]

