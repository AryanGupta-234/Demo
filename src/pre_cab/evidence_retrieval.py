"""Local evidence chunking and relevance ranking for GPT reasoning.

Large attachment text is never dumped wholesale into the model. This module selects compact passages
that are most relevant to testing, approval, rollback, environment and implementation claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .evidence import EvidenceDocument


@dataclass(frozen=True)
class EvidenceChunk:
    document_ref: str
    document_name: str
    chunk_index: int
    text: str
    score: float
    matched_terms: tuple[str, ...]


_DEFAULT_TERMS = (
    "uat",
    "user acceptance",
    "test",
    "tested",
    "expected result",
    "actual result",
    "pass",
    "approval",
    "approved",
    "customer",
    "rollback",
    "backout",
    "restore",
    "backup",
    "revert",
    "dev",
    "pre-prod",
    "preprod",
    "production",
    "downtime",
    "outage",
    "restart",
    "risk",
)


def chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    """Split text into paragraph-aware bounded chunks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if not paragraphs:
        paragraphs = [normalized.strip()] if normalized.strip() else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, max_chars - overlap)
            for start in range(0, len(paragraph), step):
                piece = paragraph[start : start + max_chars]
                if piece:
                    chunks.append(piece)
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{paragraph}".strip()
            if len(current) > max_chars:
                current = current[-max_chars:]
    if current:
        chunks.append(current)
    return chunks


def _query_terms(cr: dict[str, Any], extra_terms: Iterable[str] | None = None) -> set[str]:
    terms = set(_DEFAULT_TERMS)
    if extra_terms:
        terms.update(term.lower().strip() for term in extra_terms if term.strip())
    number = str(cr.get("Number") or cr.get("Effective number") or "").strip().lower()
    if number:
        terms.add(number)
    for field in ("Category", "Sub Category", "Configuration item"):
        value = str(cr.get(field) or "").strip().lower()
        if value and len(value) >= 3:
            terms.add(value)
    return terms


def retrieve_evidence_chunks(
    cr: dict[str, Any],
    documents: Iterable[EvidenceDocument],
    *,
    extra_terms: Iterable[str] | None = None,
    limit: int = 10,
) -> list[EvidenceChunk]:
    terms = _query_terms(cr, extra_terms)
    candidates: list[EvidenceChunk] = []

    for document in documents:
        for index, chunk in enumerate(chunk_text(document.text)):
            low = chunk.lower()
            matched = tuple(sorted(term for term in terms if term in low))
            if not matched:
                continue
            score = 0.0
            for term in matched:
                # Multi-word and CR-number matches carry more weight than generic single terms.
                weight = 2.0 if " " in term or term.startswith("chg") else 1.0
                score += weight
            # Strong result/evidence language gets a small ranking bonus.
            if "expected result" in low and "actual result" in low:
                score += 3.0
            if "approved" in low and "customer" in low:
                score += 2.0
            if any(term in low for term in ("rollback", "restore", "backup", "revert")):
                score += 1.0
            candidates.append(
                EvidenceChunk(
                    document_ref=document.ref,
                    document_name=document.name,
                    chunk_index=index,
                    text=chunk,
                    score=score,
                    matched_terms=matched,
                )
            )

    candidates.sort(key=lambda item: (item.score, len(item.matched_terms)), reverse=True)
    return candidates[:limit]
