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
