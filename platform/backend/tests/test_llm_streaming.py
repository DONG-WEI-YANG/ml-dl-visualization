import pytest

from app.api import llm_routes
from app.llm.base import LLMMessage


class FakeTutor:
    async def ask_stream(self, messages, week, topic, mode):
        yield "完整"
        yield "答案"


@pytest.mark.asyncio
async def test_two_stage_event_order(monkeypatch):
    monkeypatch.setattr(llm_routes, "build_quick_answer", lambda *args: "快速回答。")
    monkeypatch.setattr(llm_routes, "_make_tutor", lambda: FakeTutor())

    events = [event async for event in llm_routes._stream_chat_events(
        [LLMMessage(role="user", content="什麼是梯度下降？")], 4, "梯度下降", "tutor"
    )]

    assert [event["type"] for event in events] == [
        "status", "draft", "status", "refinement", "refinement", "done"
    ]
    assert events[0]["stage"] == "analyzing"
    assert events[2]["stage"] == "verifying"
    assert events[1]["elapsed_ms"] >= 0
    assert events[-1]["draft_ms"] >= 0


@pytest.mark.asyncio
async def test_draft_failure_still_streams_refinement(monkeypatch):
    def fail(*args):
        raise RuntimeError("draft failed")

    monkeypatch.setattr(llm_routes, "build_quick_answer", fail)
    monkeypatch.setattr(llm_routes, "_make_tutor", lambda: FakeTutor())
    events = [event async for event in llm_routes._stream_chat_events(
        [LLMMessage(role="user", content="問題")], 1, "課程", "tutor"
    )]
    types = [event["type"] for event in events]
    assert "draft" not in types
    assert types[-3:] == ["refinement", "refinement", "done"]
