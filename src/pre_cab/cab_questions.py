"""Deterministic CAB-question generation from validation findings."""
from __future__ import annotations

from .schemas import Finding, FindingSeverity


def predict_cab_questions(findings: list[Finding]) -> list[str]:
    questions: list[str] = []
    for finding in findings:
        if finding.severity == FindingSeverity.BLOCKING:
            if "TEST" in finding.code or "UAT" in finding.code:
                questions.append("Can the requester provide or clarify the required testing evidence?")
            elif "APPROVAL" in finding.code:
                questions.append("Can the requester provide the approval evidence that applies to this change?")
            elif "CONFLICT" in finding.code:
                questions.append("What is the disposition of the reported change conflict?")
            elif "BACKOUT" in finding.code or "ROLLBACK" in finding.code:
                questions.append("What is the recovery path if the implementation fails?")
            else:
                questions.append(finding.message)
        elif finding.severity == FindingSeverity.WARNING:
            questions.append(finding.message)
    # Stable de-duplication keeps the UI compact.
    return list(dict.fromkeys(questions))[:8]
