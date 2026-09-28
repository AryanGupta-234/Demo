"""Compatibility attachment loader.

The canonical extraction implementation lives in local_attachments.py. This module
keeps the older API while using the same document parsers and format support.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .evidence import EvidenceDocument
from .local_attachments import SUPPORTED_SUFFIXES, extract_text


def extract_attachment(path: Path) -> EvidenceDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported attachment type: {path.suffix or '<none>'}")
    text = "" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else extract_text(path)
    return EvidenceDocument(ref=str(path), name=path.name, text=text, document_type=suffix.lstrip("."))


def discover_attachments(root: Path, *, patterns: Iterable[str] | None = None) -> list[EvidenceDocument]:
    """Discover supported attachments recursively using the canonical extractor."""
    patterns = tuple(patterns or ("*.pdf", "*.xlsx", "*.xlsm", "*.docx", "*.pptx", "*.txt", "*.md", "*.csv", "*.eml", "*.json", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff"))
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    documents: list[EvidenceDocument] = []
    for path in sorted(paths):
        try:
            documents.append(extract_attachment(path))
        except (OSError, ValueError, RuntimeError):
            continue
    return documents
