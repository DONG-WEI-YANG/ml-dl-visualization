from types import SimpleNamespace

import pytest

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMMessage, LLMProviderError
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider


MESSAGES = [LLMMessage(role="user", content="hello")]


class FakeOpenAICompletions:
    def __init__(self, response):
        self.response = response

    async def create(self, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_openai_rejects_empty_content_with_sanitized_error():
    provider = OpenAIProvider(api_key="test", model="test-model")
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeOpenAICompletions(
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
                    model="test-model",
                    usage=None,
                )
            )
        )
    )

    with pytest.raises(LLMProviderError, match="empty_response"):
        await provider.chat(MESSAGES)


@pytest.mark.asyncio
async def test_openai_accepts_valid_content_when_usage_is_absent():
    provider = OpenAIProvider(api_key="test", model="test-model")
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeOpenAICompletions(
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                    model="resolved-model",
                    usage=None,
                )
            )
        )
    )

    response = await provider.chat(MESSAGES)

    assert response.content == "answer"
    assert response.model == "resolved-model"
    assert response.usage is None


class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response

    async def create(self, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_anthropic_rejects_response_without_text_block():
    provider = AnthropicProvider(api_key="test", model="test-model")
    provider.client = SimpleNamespace(
        messages=FakeAnthropicMessages(
            SimpleNamespace(content=[], model="test-model", usage=None)
        )
    )

    with pytest.raises(LLMProviderError, match="empty_response"):
        await provider.chat(MESSAGES)


class FakeHTTPResponse:
    def __init__(self, data=None, lines=(), error=None):
        self._data = data
        self._lines = lines
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._data

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return False


class FakeHTTPClient:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return self.response

    def stream(self, *args, **kwargs):
        return FakeStreamContext(self.response)


@pytest.mark.asyncio
async def test_ollama_checks_http_status_before_reading_body(monkeypatch):
    fake = FakeHTTPResponse(error=RuntimeError("raw upstream secret"))
    monkeypatch.setattr(
        "app.llm.ollama_provider.httpx.AsyncClient",
        lambda: FakeHTTPClient(fake),
    )
    provider = OllamaProvider(model="test-model")

    with pytest.raises(LLMProviderError, match="request_failed") as error:
        await provider.chat(MESSAGES)

    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_ollama_stream_skips_malformed_frames_and_yields_valid_text(monkeypatch):
    fake = FakeHTTPResponse(
        lines=(
            "not-json",
            '{"message":{"content":"first"}}',
            '{"done":true}',
            '{"message":{"content":"second"}}',
        )
    )
    monkeypatch.setattr(
        "app.llm.ollama_provider.httpx.AsyncClient",
        lambda: FakeHTTPClient(fake),
    )
    provider = OllamaProvider(model="test-model")

    chunks = [chunk async for chunk in provider.stream(MESSAGES)]

    assert chunks == ["first", "second"]
