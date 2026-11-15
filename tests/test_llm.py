import json

import pytest

from src.answering import llm
from src.answering.llm import (
    LLMGenerationError,
    LLMUnavailableError,
    OllamaClient,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_available_requires_configured_model(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: FakeResponse({"models": [{"name": "other:3b"}]}),
    )
    client = OllamaClient(model="qwen3:4b")

    assert client.available() is False
    assert "not installed" in client.last_error


def test_generate_json_rejects_malformed_model_json(monkeypatch):
    responses = iter([
        FakeResponse({"models": [{"name": "test:1b"}]}),
        FakeResponse({"response": "not-json"}),
    ])
    monkeypatch.setattr(llm, "urlopen", lambda request, timeout: next(responses))
    client = OllamaClient(model="test:1b")

    with pytest.raises(LLMGenerationError, match="malformed JSON"):
        client.generate_json("prompt")


def test_generate_json_preserves_context_length_error(monkeypatch):
    responses = iter([
        FakeResponse({"models": [{"name": "test:1b"}]}),
        FakeResponse({"error": "context length exceeded"}),
    ])
    monkeypatch.setattr(llm, "urlopen", lambda request, timeout: next(responses))
    client = OllamaClient(model="test:1b")

    with pytest.raises(LLMGenerationError, match="context-length"):
        client.generate_json("prompt")


def test_generate_json_fails_cleanly_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: FakeResponse({"models": []}),
    )
    client = OllamaClient(model="missing:1b")

    with pytest.raises(LLMUnavailableError, match="not installed"):
        client.generate_json("prompt")