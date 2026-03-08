from app.llm.base import LLMMessage, LLMResponse


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
