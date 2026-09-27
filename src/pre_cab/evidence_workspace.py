"""Dynamic CR evidence workspace and field-aware document discovery.

Creates a local <root>/<CR number> workspace for each incoming CR and optionally copies
caller-supplied evidence files into it. Files are never fetched from the internet.
Classification is filename/content based and is only used to prioritize document review.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .evidence import EvidenceDocument
from .local_attachments import SUPPORTED_SUFFIXES, load_attachments_for_cr

EVIDENCE_TYPES: dict[str, tuple[str, ...]] = {
    "customer_approval": ("customer approval", "customer_approval", "customer-approval", "approval", "client approval"),
    "uat": ("uat", "user acceptance", "acceptance signoff", "uat signoff"),
    "qa_signoff": ("tcs qa", "qa signoff", "quality assurance", "qa approval"),
    "test_results": ("test result", "test execution", "test evidence", "test report", "execution report", "validation"),
    "lower_environment": ("lower environment", "pre-prod", "preprod", "sit", "staging", "lower env"),
    "rollback": ("rollback", "backout", "recovery", "restore"),
    "implementation": ("implementation", "deployment", "change plan", "runbook"),
}

def safe_cr_dir(root: str | Path, cr_number: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    safe_number = re.sub(r"[^A-Za-z0-9._-]", "_", cr_number.strip()) or "UNKNOWN_CR"
    path = root_path / safe_number
    path.mkdir(parents=True, exist_ok=True)
    return path

def stage_files(root: str | Path, cr_number: str, files: Iterable[str | Path]) -> Path:
    """Create the CR folder and copy supported local evidence into it."""
    folder = safe_cr_dir(root, cr_number)
    for source in files:
        src = Path(source).expanduser()
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        destination = folder / src.name
        if src.resolve() != destination.resolve():
            shutil.copy2(src, destination)
    return folder

def classify_document(document: EvidenceDocument) -> dict[str, Any]:
    haystack = f"{document.name}\n{document.text[:12000]}".lower()
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for kind, terms in EVIDENCE_TYPES.items():
        hits = [term for term in terms if term in haystack]
        if hits:
            scores[kind] = len(hits)
            matched[kind] = hits
    best_kind = max(scores, key=scores.get) if scores else "other"
    return {
        "type": best_kind,
        "scores": scores,
        "matched_terms": matched.get(best_kind, []),
        "document": document.name,
        "ref": document.ref,
    }

def load_and_classify(root: str | Path, cr_number: str) -> tuple[list[EvidenceDocument], list[dict[str, Any]]]:
    documents = load_attachments_for_cr(Path(root), cr_number)
    classifications = [classify_document(document) for document in documents]
    for document, classification in zip(documents, classifications):
        document.metadata["evidence_classification"] = classification
    return documents, classifications
