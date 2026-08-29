import pytest

from app.llm.base import LLMMessage, LLMResponse
from app.llm.tutor import AITutor


def test_llm_message_model():
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_llm_response_model():
    resp = LLMResponse(content="Hi", model="test")
    assert resp.content == "Hi"
    assert resp.usage is None


def test_llm_response_with_usage():
    resp = LLMResponse(content="Hi", model="test", usage={"input": 10, "output": 5})
    assert resp.usage["input"] == 10


class CaptureProvider:
    def __init__(self):
        self.system = ""

    async def chat(self, messages, system=""):
        self.system = system
        return LLMResponse(content="ok", model="capture")

    async def stream(self, messages, system=""):
        self.system = system
        yield "ok"


@pytest.mark.asyncio
async def test_tutor_ask_includes_requested_student_context(monkeypatch):
    monkeypatch.setattr("app.llm.tutor.retrieve_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "app.llm.tutor._get_student_context",
        lambda student_id: f"\n學生識別：{student_id}",
    )
    provider = CaptureProvider()

    await AITutor(provider, use_rag=False).ask(
        [LLMMessage(role="user", content="問題")],
        week=1,
        topic="基礎",
        student_id="42",
    )

    assert "學生識別：42" in provider.system


@pytest.mark.asyncio
async def test_tutor_stream_includes_requested_student_context(monkeypatch):
    monkeypatch.setattr("app.llm.tutor.retrieve_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "app.llm.tutor._get_student_context",
        lambda student_id: f"\n學生識別：{student_id}",
    )
    provider = CaptureProvider()

    chunks = [chunk async for chunk in AITutor(provider, use_rag=False).ask_stream(
        [LLMMessage(role="user", content="問題")],
        week=1,
        topic="基礎",
        student_id="84",
    )]

    assert chunks == ["ok"]
    assert "學生識別：84" in provider.system
