import json

import pytest

from pre_cab.servicenow_readonly import ReadOnlyServiceNowClient, ServiceNowCredentials


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = _Headers({"Content-Length": str(len(payload))})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if not self.payload:
            return b""
        if size < 0:
            chunk, self.payload = self.payload, b""
        else:
            chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


def test_get_uses_get_only_and_returns_json(monkeypatch):
    captured = {}
    payload = json.dumps({"result": [{"number": "CHG001"}]}).encode()

    def fake_urlopen(request, timeout):
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response(payload)

    monkeypatch.setattr("pre_cab.servicenow_readonly.urlopen", fake_urlopen)
    client = ReadOnlyServiceNowClient(
        ServiceNowCredentials("https://example.service-now.com", "user", "secret"),
        max_retries=0,
    )

    result = client.get_change("CHG001")

    assert result["number"] == "CHG001"
    assert captured["method"] == "GET"
    assert captured["timeout"] == 30.0
    assert captured["auth"].startswith("Basic ")
    assert "secret" not in captured["url"]


def test_attachment_size_limit_rejects_oversized_response(monkeypatch):
    payload = b"x" * 11
    monkeypatch.setattr(
        "pre_cab.servicenow_readonly.urlopen",
        lambda request, timeout: _Response(payload),
    )
    client = ReadOnlyServiceNowClient(
        ServiceNowCredentials("https://example.service-now.com", "user", "secret"),
        max_retries=0,
        max_attachment_bytes=10,
    )

    with pytest.raises(ValueError, match="size limit"):
        client.download_attachment("abc")



def test_servicenow_image_attachment_uses_local_ocr(monkeypatch):
    monkeypatch.setattr(
        "pre_cab.servicenow_readonly.analyze_image_evidence",
        lambda path: (
            "CHG001 image OCR text",
            {"ocr_executed": True, "ocr_engine": "fake", "requires_vision": True, "vision_executed": False},
        ),
    )
    client = ReadOnlyServiceNowClient(
        ServiceNowCredentials("https://example.service-now.com", "user", "secret"),
        max_retries=0,
    )

    text, metadata = client._extract_attachment_text("CHG001_screenshot.png", b"image")
    assert "CHG001 image OCR text" in text
    assert metadata["ocr_executed"] is True
    assert metadata["requires_vision"] is True


def test_servicenow_scanned_pdf_runs_ocr_when_native_text_is_empty(monkeypatch):
    monkeypatch.setattr(
        "pre_cab.servicenow_readonly.extract_text",
        lambda path: "",
    )
    monkeypatch.setattr(
        "pre_cab.servicenow_readonly.ocr_pdf",
        lambda path: (
            "[OCR PAGE 1]\nCHG001 TEST RESULT PASS",
            {"ocr_executed": True, "ocr_engine": "fake", "ocr_confidence": 0.91},
        ),
    )
    client = ReadOnlyServiceNowClient(
        ServiceNowCredentials("https://example.service-now.com", "user", "secret"),
        max_retries=0,
    )

    text, metadata = client._extract_attachment_text("CHG001_scan.pdf", b"pdf")
    assert "TEST RESULT PASS" in text
    assert metadata["ocr_executed"] is True
    assert metadata["ocr_confidence"] == 0.91
