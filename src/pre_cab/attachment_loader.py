"""Local attachment discovery and extraction for Stage 2.

The loader is intentionally filesystem-only: it does not upload documents anywhere. It produces
EvidenceDocument records consumed by the existing evidence validator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .evidence import EvidenceDocument


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse PDF attachments") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse XLSX attachments") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    chunks: list[str] = []
    for sheet in workbook.worksheets:
        chunks.append(f"[SHEET {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            values = [str(value) for value in row if value is not None]
            if values:
                chunks.append(" | ".join(values))
    return "\n".join(chunks)


def extract_attachment(path: Path) -> EvidenceDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(path)
        doc_type = "pdf"
    elif suffix in {".xlsx", ".xlsm"}:
        text = _extract_xlsx(path)
        doc_type = "xlsx"
    elif suffix in {".txt", ".md", ".csv", ".log", ".eml"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_type = suffix.lstrip(".")
    else:
        raise ValueError(f"Unsupported attachment type: {path.suffix or '<none>'}")
    return EvidenceDocument(ref=str(path), name=path.name, text=text, document_type=doc_type)


def discover_attachments(root: Path, *, patterns: Iterable[str] | None = None) -> list[EvidenceDocument]:
    """Discover supported attachments recursively and return extracted documents."""
    patterns = tuple(patterns or ("*.pdf", "*.xlsx", "*.xlsm", "*.txt", "*.md", "*.csv", "*.log", "*.eml"))
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    documents: list[EvidenceDocument] = []
    for path in sorted(paths):
        try:
            documents.append(extract_attachment(path))
        except (OSError, ValueError, RuntimeError):
            # One unreadable or unsupported file should not prevent the rest of the evidence
            # corpus from being analyzed. The caller can surface skipped paths separately.
            continue
    return documents
