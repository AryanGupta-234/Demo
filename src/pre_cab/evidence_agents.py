"""Specialist agents for document-backed Pre-CAB evidence verification."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evidence import EvidenceDocument


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    low = _norm(text)
    return any(term.lower() in low for term in terms)


def _cr_match(cr: dict[str, Any], doc: EvidenceDocument) -> bool:
    number = _norm(cr.get("Number") or cr.get("Effective number"))
    if not number:
        return True
    return number in _norm(doc.name) or number in _norm(doc.text)


def _status(text: str) -> str:
    low = _norm(text)
    if re.search(r"\b(rejected|declined|failed|not approved|not passed|pending approval)\b", low):
        return "negative"
    if re.search(r"\b(approved|accepted|passed|completed|successful|success)\b", low):
        return "positive"
    return "unclear"


class DocumentEvidenceAgent:
    """Evaluate extracted document content against one evidence purpose."""

    def __init__(self, purpose: str, terms: tuple[str, ...]) -> None:
        self.purpose = purpose
        self.terms = terms

    def run(self, cr: dict[str, Any], documents: list[EvidenceDocument]) -> dict[str, Any]:
        candidates = [
            d for d in documents
            if _contains(f"{d.name}\n{d.text[:20000]}", self.terms) and _cr_match(cr, d)
        ]
        return {
            "purpose": self.purpose,
            "candidates": [
                {
                    "name": d.name,
                    "ref": d.ref,
                    "document_type": d.document_type,
                    "bytes": d.metadata.get("bytes"),
                    "status": _status(d.text),
                    "text_chars": len(d.text),
                }
                for d in candidates
            ],
            "positive_count": sum(_status(d.text) == "positive" for d in candidates),
            "negative_count": sum(_status(d.text) == "negative" for d in candidates),
            "ambiguous_count": sum(_status(d.text) == "unclear" for d in candidates),
        }


class CustomerApprovalAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__("customer_approval", (
            "customer approval", "customer approved", "approved by customer",
            "customer signoff", "client approval", "client approved",
        ))


class UATAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__("uat", (
            "uat", "user acceptance", "acceptance signoff", "business signoff",
            "business acceptance", "uat approved",
        ))


class QASignoffAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__("qa_signoff", (
            "tcs qa", "qa signoff", "quality assurance", "qa approval", "qa completed",
        ))


class TestEvidenceAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__("test_results", (
            "test result", "test execution", "test evidence", "expected result",
            "actual result", "test case", "execution report", "validation result",
        ))


class LowerEnvironmentAgent(DocumentEvidenceAgent):
    def __init__(self) -> None:
        super().__init__("lower_environment", (
            "lower environment", "pre-prod", "preprod", "sit", "staging",
            "lower env", "system integration test",
        ))


DEFAULT_DOCUMENT_AGENTS = (
    CustomerApprovalAgent(),
    UATAgent(),
    QASignoffAgent(),
    TestEvidenceAgent(),
    LowerEnvironmentAgent(),
)
