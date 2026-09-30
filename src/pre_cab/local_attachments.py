"""Safe local CR attachment discovery, inventory and extraction.

The upstream ingestion layer is expected to place files under <root>/<CR number>.
This module never calls ServiceNow or follows evidence belonging to another CR.
It inventories every file, while only extracting supported document formats.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .evidence import EvidenceDocument
from .ocr import OCRUnavailable, analyze_image_evidence, ocr_pdf

TEXT_SUFFIXES = {
    ".pdf", ".xlsx", ".xlsm", ".docx", ".pptx",
    ".txt", ".md", ".csv", ".eml", ".json",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES
DEFAULT_MAX_FILE_BYTES = 75 * 1024 * 1024


def _max_file_bytes() -> int:
    raw = os.getenv("PRE_CAB_MAX_ATTACHMENT_BYTES", str(DEFAULT_MAX_FILE_BYTES))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_FILE_BYTES


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


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse DOCX attachments") from exc
    document = Document(str(path))
    chunks = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        chunks.append("[TABLE]")
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                chunks.append(" | ".join(values))
    return "\n".join(chunks)


def _read_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("Install the 'docs' extra to parse PPTX attachments") from exc
    presentation = Presentation(str(path))
    chunks: list[str] = []
    for index, slide in enumerate(presentation.slides, 1):
        chunks.append(f"[SLIDE {index}]")
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if text and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".xlsx", ".xlsm"}:
        return _read_xlsx(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pptx":
        return _read_pptx(path)
    if suffix in {".txt", ".md", ".csv", ".eml", ".json"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _cr_directory(root: Path, cr_number: str) -> Path | None:
    needle = cr_number.strip()
    if not needle:
        return None

    root_resolved = root.resolve()
    direct = root / needle
    if direct.is_dir() and not direct.is_symlink():
        resolved = direct.resolve()
        if _inside(resolved, root_resolved):
            return resolved

    for candidate in root.rglob(needle):
        if not candidate.is_dir() or candidate.is_symlink() or candidate.name != needle:
            continue
        resolved = candidate.resolve()
        if _inside(resolved, root_resolved):
            return resolved
    return None


def _root_attachments(root: Path, cr_number: str) -> list[Path]:
    needle = cr_number.strip()
    if not needle:
        return []
    matches: list[Path] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = path.stem
        if stem == needle or stem.startswith(f"{needle}_") or stem.startswith(f"{needle}-"):
            matches.append(path)
    return sorted(matches)


def discover_attachments(root: Path, cr_number: str) -> list[Path]:
    """Find supported files strictly inside the CR workspace."""
    if not root.exists() or not root.is_dir():
        return []

    cr_dir = _cr_directory(root, cr_number)
    if cr_dir is not None:
        files: list[Path] = []
        for path in cr_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            # Avoid symlinks escaping the CR workspace.
            if path.is_symlink() or not _inside(path, cr_dir):
                continue
            files.append(path)
        return sorted(files)

    return _root_attachments(root, cr_number)


def inventory_workspace(root: Path, cr_number: str) -> list[dict[str, object]]:
    """Inventory every file in the CR workspace, including unsupported/unreadable files."""
    cr_dir = _cr_directory(root, cr_number)
    if cr_dir is None:
        paths = _root_attachments(root, cr_number)
    else:
        paths = [p for p in cr_dir.rglob("*") if p.is_file() and not p.is_symlink() and _inside(p, cr_dir)]

    inventory: list[dict[str, object]] = []
    for path in sorted(paths):
        try:
            size = path.stat().st_size
        except OSError as exc:
            inventory.append({"name": path.name, "path": str(path), "status": "stat_error", "error": str(exc)})
            continue
        inventory.append({
            "name": path.name,
            "path": str(path),
            "suffix": path.suffix.lower(),
            "bytes": size,
            "supported": path.suffix.lower() in SUPPORTED_SUFFIXES,
            "status": "too_large" if size > _max_file_bytes() else ("supported" if path.suffix.lower() in SUPPORTED_SUFFIXES else "unsupported"),
        })
    return inventory


def load_attachments_for_cr(root: Path, cr_number: str) -> list[EvidenceDocument]:
    documents: list[EvidenceDocument] = []
    for path in discover_attachments(root, cr_number):
        suffix = path.suffix.lower()
        metadata = {
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "requires_vision": suffix in IMAGE_SUFFIXES,
            "ocr_executed": False,
            "vision_executed": False,
            "extraction_error": None,
        }

        if suffix in IMAGE_SUFFIXES:
            text, derived = analyze_image_evidence(path)
            metadata.update(derived)
            if not text:
                metadata["extraction_error"] = (
                    metadata.get("vision_error")
                    or metadata.get("ocr_error")
                    or "No OCR/vision text was produced."
                )
        elif metadata["bytes"] > _max_file_bytes():
            text = ""
            metadata["extraction_error"] = f"File exceeds PRE_CAB_MAX_ATTACHMENT_BYTES={_max_file_bytes()}."
        else:
            try:
                text = extract_text(path)
                if suffix == ".pdf" and not text.strip():
                    # A textless PDF is a likely scanned document. Render it locally and
                    # execute OCR without replacing the original document evidence.
                    ocr_text, ocr_meta = ocr_pdf(path)
                    metadata.update(ocr_meta)
                    if ocr_text:
                        text = ocr_text
                    else:
                        metadata["extraction_error"] = "No extractable text found after PDF OCR."
                elif not text.strip():
                    metadata["extraction_error"] = "No extractable text found; document may be empty."
            except OCRUnavailable as exc:
                text = ""
                metadata["extraction_error"] = str(exc)
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

