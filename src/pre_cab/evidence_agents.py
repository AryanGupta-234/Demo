"""Specialist agents for document-backed Pre-CAB evidence verification.

These agents operate on locally extracted EvidenceDocument objects. They do not invent approval
or testing evidence; they convert extracted text into structured verification signals for the
final reasoning stage.
"""
from __future__ import annotations

import re
from typing import Any

from .evidence import EvidenceDocument
from .schemas import Finding, FindingSeverity


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    low = _norm(text)
    return any(term in low for term in terms)


def _cr_match(cr: dict[str, Any], doc: EvidenceDocument) -> bool:
    number = _norm(cr.get("Number") or cr.get("Effective number"))
    if not number:
        return True
    return number in _norm(doc.name) or number in _norm(doc.text)


def _extract_status(text: str) -> str:
    low = _norm(text)
    if re.search(r"\b(rejected|declined|failed|not approved|not passed)\b", low):
        return "negative"
    if re.search(r"\b(approved|accepted|passed|complete|completed|successful|success)\b", low):
        return "positive"
    return "unclear"


class DocumentEvidenceAgent:
    """Evaluate document content against a single evidence purpose."""

    def __init__(self, purpose: str, terms: tuple[str, ...]) -> None:
        self.purpose = purpose
        self.terms = terms

    def run(self, cr: dict[str, Any], documents: list[EvidenceDocument]) -> dict[str, Any]:
        candidates = [
            d for d in documents
            if _contains(f"{d.name}\n{d.text[:20000]}", self.terms) and _cr_match(cr, d)
        ]
        statuses = [_extract_status(d.text) for d in candidates]
        positive = any(s == "positive" for s in statuses)
        negative = any(s == "negative" for s in statuses)
        return {
            "purpose": self.purpose,
            "candidate_documents": [
                {
                    "name": d.name,
                    "ref": d.ref,
                    "document_type": d.document_type,
                    "text_chars": len(d.text),
                    "status": _extract_status(d.text),
                }
                for d in candidates
            ],
            "verified": bool(candidates) and positive and not negative,
            "negative": negative,
            "ambiguous": bool(candidates) and not positive and not negative,
        }


class CustomerApprovalAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__(
            "customer_approval",
            ("customer approval", "customer approved", "approved by customer", "customer signoff"),
        )


class UATAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__(
            "uat",
            ("uat", "user acceptance", "acceptance signoff", "business signoff"),
        )


class QASignoffAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__(
            "qa_signoff",
            ("tcs qa", "qa signoff", "quality assurance", "qa approval"),
        )


class TestEvidenceAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__(
            "test_results",
            ("test result", "test execution", "test evidence", "expected result", "actual result", "test case"),
        )


class LowerEnvironmentAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__(
            "lower_environment",
            ("lower environment", "pre-prod", "preprod", "sit", "staging", "lower env"),
        )


DEFAULT_DOCUMENT_AGENTS = (
    CustomerApprovalAgent(),
    UATAgent(),
    QASignoffAgent(),
    TestEvidenceAgent(),
    LowerEnvironmentAgent(),
)
