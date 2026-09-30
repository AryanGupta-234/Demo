"""Local OCR and optional Ollama vision execution for evidence files.

OCR is intentionally local-first:
- Tesseract + pytesseract handles raster images.
- PyMuPDF renders scanned PDF pages before OCR.
- An optional Ollama vision model can inspect image bytes and return a
  clearly-marked derived description for neural reasoning.

No cloud OCR service is required.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OCRUnavailable(RuntimeError):
    """Raised when OCR dependencies or the local OCR engine are unavailable."""


class VisionUnavailable(RuntimeError):
    """Raised when the configured local vision provider is unavailable."""


def _tesseract_path() -> str | None:
    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        return configured if Path(configured).exists() else None

    found = shutil.which("tesseract")
    if found:
        return found

    if os.name == "nt":
        common = (
            Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
            Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
            Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        )
        for candidate in common:
            if candidate.exists():
                return str(candidate)
    return None


def tesseract_available() -> bool:
    return _tesseract_path() is not None


def ocr_image(path: Path) -> tuple[str, dict[str, Any]]:
    """Run Tesseract OCR over one raster image."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailable("Install the ocr extra to enable local OCR (pytesseract + Pillow).") from exc

    command = _tesseract_path()
    if not command:
        raise OCRUnavailable(
            "Tesseract OCR engine is not installed. Install Tesseract or set TESSERACT_CMD."
        )
    pytesseract.pytesseract.tesseract_cmd = command

    try:
        image = Image.open(path)
        image.load()
        text = pytesseract.image_to_string(image).strip()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = []
        for raw in data.get("conf", []):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                confidences.append(value)
        confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else None
    except Exception as exc:
        raise OCRUnavailable(f"OCR failed for {path.name}: {type(exc).__name__}: {exc}") from exc

    return text, {
        "ocr_engine": "tesseract",
        "ocr_executed": True,
        "ocr_confidence": round(confidence, 3) if confidence is not None else None,
        "ocr_text_chars": len(text),
    }


def ocr_pdf(path: Path, *, max_pages: int | None = None, dpi: int = 160) -> tuple[str, dict[str, Any]]:
    """Render PDF pages with PyMuPDF and OCR pages that contain little/no text."""
    try:
        import fitz
    except ImportError as exc:
        raise OCRUnavailable("Install the ocr extra to enable scanned-PDF OCR (PyMuPDF).") from exc

    configured_pages = os.getenv("PRE_CAB_OCR_MAX_PDF_PAGES", "").strip()
    if max_pages is None:
        try:
            max_pages = max(1, int(configured_pages)) if configured_pages else 20
        except ValueError:
            max_pages = 20

    try:
        document = fitz.open(str(path))
        page_count = min(len(document), max_pages)
        parts: list[str] = []
        page_confidences: list[float] = []
        pages_ocrd = 0
        for index in range(page_count):
            page = document.load_page(index)
            native_text = (page.get_text("text") or "").strip()
            if native_text:
                continue
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image_bytes = pix.tobytes("png")
            temp_path = path.with_suffix(f".ocr-page-{index}.png")
            temp_path.write_bytes(image_bytes)
            try:
                text, metadata = ocr_image(temp_path)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            if text:
                parts.append(f"[OCR PAGE {index + 1}]\n{text}")
            if metadata.get("ocr_confidence") is not None:
                page_confidences.append(float(metadata["ocr_confidence"]))
            pages_ocrd += 1
        document.close()
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"Scanned-PDF OCR failed for {path.name}: {type(exc).__name__}: {exc}") from exc

    confidence = sum(page_confidences) / len(page_confidences) if page_confidences else None
    return "\n".join(parts).strip(), {
        "ocr_engine": "tesseract+pymupdf",
        "ocr_executed": pages_ocrd > 0,
        "ocr_pages_processed": pages_ocrd,
        "ocr_pages_limit": page_count,
        "ocr_confidence": round(confidence, 3) if confidence is not None else None,
    }


def vision_enabled() -> bool:
    return bool(os.getenv("PRE_CAB_VISION_MODEL", "").strip())


def ollama_vision_analyze(path: Path, *, prompt: str) -> tuple[str, dict[str, Any]]:
    """Inspect an image with a local Ollama vision-capable model.

    The REST API expects base64 image data in the message's images array.
    """
    model = os.getenv("PRE_CAB_VISION_MODEL", "").strip()
    if not model:
        raise VisionUnavailable("PRE_CAB_VISION_MODEL is not configured.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
        }],
        "stream": False,
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=float(os.getenv("PRE_CAB_VISION_TIMEOUT", "120"))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise VisionUnavailable(f"Ollama vision request failed: {type(exc).__name__}: {exc}") from exc

    message = body.get("message") if isinstance(body, dict) else None
    text = str(message.get("content") or "").strip() if isinstance(message, dict) else ""
    if not text:
        raise VisionUnavailable("Ollama vision returned no text.")

    return text, {
        "vision_executed": True,
        "vision_provider": "ollama",
        "vision_model": model,
        "vision_role": "derived_image_description_not_direct_evidence",
    }


def analyze_image_evidence(path: Path) -> tuple[str, dict[str, Any]]:
    """Run OCR, then optional vision, and return a combined derived evidence view."""
    metadata: dict[str, Any] = {
        "ocr_executed": False,
        "vision_executed": False,
        "requires_vision": True,
    }

    ocr_text = ""
    try:
        ocr_text, ocr_meta = ocr_image(path)
        metadata.update(ocr_meta)
    except OCRUnavailable as exc:
        metadata["ocr_error"] = str(exc)

    vision_text = ""
    if vision_enabled():
        try:
            vision_text, vision_meta = ollama_vision_analyze(
                path,
                prompt=(
                    "Inspect this change-management evidence image. Extract only information visibly present "
                    "in the image that could matter to a Pre-CAB review: CR/change number, document type, "
                    "approval/signoff status, test execution/result status, environment, rollback/recovery, "
                    "dates, names/roles if clearly shown, and contradictions. Do not guess missing values. "
                    "Return a concise factual description."
                ),
            )
            metadata.update(vision_meta)
        except VisionUnavailable as exc:
            metadata["vision_error"] = str(exc)

    sections: list[str] = []
    if ocr_text:
        sections.append("[OCR EXTRACTED TEXT]\n" + ocr_text)
    if vision_text:
        sections.append("[VISION DERIVED DESCRIPTION — NOT DIRECT EVIDENCE]\n" + vision_text)
    combined = "\n\n".join(sections).strip()
    metadata["derived_text_chars"] = len(combined)
    return combined, metadata
