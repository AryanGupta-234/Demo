"""Local attachment discovery for offline/private benchmark and demo runs.

Files remain on the caller's machine. Supported text/tabular documents are extracted locally. Image
attachments are retained as explicit unparsed evidence so the validator never silently ignores them.
"""
from __future__ import annotations

from pathlib import Path

from .evidence import EvidenceDocument

TEXT_SUFFIXES = {".pdf", ".xlsx", ".xlsm", ".txt", ".md", ".csv", ".eml", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES


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
    """Find supported files associated by CR number in the path; never guess unrelated evidence."""
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
        suffix = path.suffix.lower()
        metadata = {
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "requires_vision": suffix in IMAGE_SUFFIXES,
            "extraction_error": None,
        }
        if suffix in IMAGE_SUFFIXES:
            text = ""
        else:
            try:
                text = extract_text(path)
                if not text.strip():
                    metadata["extraction_error"] = "No extractable text found; document may be scanned/image-only or empty."
            except (OSError, RuntimeError, ValueError) as exc:
                text = ""
                metadata["extraction_error"] = f"{type(exc).__name__}: {exc}"
        documents.append(
            EvidenceDocument(
                ref=str(path),
                name=path.name,
                text=text,
                document_type=suffix.lstrip(".") or "unknown",
                metadata=metadata,
            )
        )
    return documents
