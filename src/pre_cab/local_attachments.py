"""Local attachment discovery for offline/private benchmark and demo runs.

Files remain on the caller's machine. This module discovers likely attachments for a CR and converts
supported formats into EvidenceDocument records consumed by Stage 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .evidence import EvidenceDocument

SUPPORTED_SUFFIXES = {".pdf", ".xlsx", ".xlsm", ".txt", ".md", ".csv", ".eml", ".json"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse PDF attachments") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse XLSX attachments") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        lines.append(f"[Sheet: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            values = [str(value) for value in row if value not in (None, "")]
            if values:
                lines.append(" | ".join(values))
    return "\n".join(lines)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".xlsx", ".xlsm"}:
        return _read_xlsx(path)
    if suffix in {".txt", ".md", ".csv", ".eml", ".json"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def discover_attachments(root: Path, cr_number: str) -> list[Path]:
    """Find supported files likely belonging to a CR.

    Primary association is the CR number in file/folder names. If none are named that way, callers can
    still pass explicit documents to the main pipeline; this function deliberately avoids guessing.
    """
    if not root.exists():
        return []
    needle = cr_number.lower().strip()
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = str(path.relative_to(root)).lower()
        if needle and needle in relative:
            matches.append(path)
    return sorted(matches)


def load_attachments_for_cr(root: Path, cr_number: str) -> list[EvidenceDocument]:
    documents: list[EvidenceDocument] = []
    for path in discover_attachments(root, cr_number):
        try:
            text = extract_text(path)
        except Exception as exc:
            text = f"[EXTRACTION_ERROR] {type(exc).__name__}: {exc}"
        documents.append(
            EvidenceDocument(
                ref=str(path),
                name=path.name,
                text=text,
                document_type=path.suffix.lower().lstrip(".") or "unknown",
                metadata={"local_path": str(path), "bytes": path.stat().st_size},
            )
        )
    return documents
