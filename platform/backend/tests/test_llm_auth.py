"""LLM endpoints must require authentication."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _token():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


def test_chat_rejects_anonymous():
    resp = client.post("/api/llm/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code in (401, 403)


def test_chat_accepts_authenticated():
    resp = client.post(
        "/api/llm/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert resp.status_code == 200


def test_ws_rejects_missing_token():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/llm/ws/chat"):
            pass
    assert exc_info.value.code == 4401


def test_ws_rejects_invalid_token():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/llm/ws/chat?token=bad.token.here"):
            pass
    assert exc_info.value.code == 4401
