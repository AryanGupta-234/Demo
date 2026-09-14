import json
from unittest.mock import MagicMock, patch

import pytest

from pre_cab.ollama_provider import OllamaProvider


def _fake_response(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = json.dumps(payload).encode("utf-8")
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    return mock


def test_generate_sends_expected_payload_and_parses_response():
    provider = OllamaProvider(model_id="qwen2.5:7b-instruct", base_url="http://localhost:11434")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_response({"message": {"role": "assistant", "content": '{"prediction": "PASS"}'}})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        response = provider.generate(system="sys", user="usr", temperature=0.2)

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "qwen2.5:7b-instruct"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.2
    assert response.text == '{"prediction": "PASS"}'
    assert response.model == "qwen2.5:7b-instruct"


def test_connection_error_gives_actionable_message():
    provider = OllamaProvider()

    def fake_urlopen(request, timeout):
        from urllib.error import URLError
        raise URLError("Connection refused")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(RuntimeError, match="ollama serve"):
            provider.generate(system="sys", user="usr")


def test_model_name_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    provider = OllamaProvider()
    assert provider.model_name == "qwen2.5:7b-instruct"
    assert provider.base_url == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.5:11434/")
    provider2 = OllamaProvider()
    assert provider2.model_name == "llama3.1:8b"
    assert provider2.base_url == "http://192.168.1.5:11434"  # trailing slash stripped
