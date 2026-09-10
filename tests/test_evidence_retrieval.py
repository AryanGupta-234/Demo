from __future__ import annotations

from pre_cab.evidence import EvidenceDocument
from pre_cab.evidence_retrieval import chunk_text, retrieve_evidence_chunks


def test_chunk_text_bounds_large_document() -> None:
    text = "A" * 5000
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_retrieval_prefers_execution_and_customer_approval_evidence() -> None:
    cr = {
        "Number": "CHG-DEMO-900",
        "Category": "Payments",
        "Configuration item": "Wallet Service",
    }
    documents = [
        EvidenceDocument("generic", "generic.txt", "General notes about the system."),
        EvidenceDocument(
            "uat",
            "uat.txt",
            "CHG-DEMO-900 UAT test case. Expected result: success. Actual result: success. PASS.",
        ),
        EvidenceDocument(
            "approval",
            "approval.txt",
            "CHG-DEMO-900 customer approval received. Customer approved the production change.",
        ),
    ]
    chunks = retrieve_evidence_chunks(cr, documents, limit=3)
    refs = [chunk.document_ref for chunk in chunks]
    assert "uat" in refs
    assert "approval" in refs
    assert "generic" not in refs
    assert chunks[0].score > 0
