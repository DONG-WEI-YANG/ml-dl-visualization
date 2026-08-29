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


def test_chat_accepts_authenticated(monkeypatch):
    captured = {}

    async def fake_ask(self, messages, week, topic, mode="tutor", student_id=None):
        captured["student_id"] = student_id
        from app.llm.base import LLMResponse
        return LLMResponse(content="ok", model="fake")

    monkeypatch.setattr("app.api.llm_routes.AITutor.ask", fake_ask)
    token = _token()
    resp = client.post(
        "/api/llm/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert captured["student_id"] == str(me["id"])


def test_ws_forwards_authenticated_identity_to_stream(monkeypatch):
    captured = {}

    async def fake_events(messages, week, topic, mode, student_id=None):
        captured["student_id"] = student_id
        yield {"type": "done", "elapsed_ms": 1, "draft_ms": None}

    monkeypatch.setattr("app.api.llm_routes._stream_chat_events", fake_events)
    token = _token()
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()

    with client.websocket_connect(f"/api/llm/ws/chat?token={token}") as websocket:
        websocket.send_json({"messages": [{"role": "user", "content": "hi"}]})
        assert websocket.receive_json()["type"] == "done"

    assert captured["student_id"] == str(me["id"])


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


def test_ws_rejects_soft_deleted_user():
    import pytest
    from starlette.websockets import WebSocketDisconnect

    admin_headers = {"Authorization": f"Bearer {_token()}"}
    client.post(
        "/api/auth/register",
        json={"username": "ws_soft_del", "password": "somepass1", "role": "student"},
        headers=admin_headers,
    )
    login = client.post(
        "/api/auth/login", json={"username": "ws_soft_del", "password": "somepass1"}
    )
    student_token = login.json()["access_token"]
    student_id = login.json()["user"]["id"]
    client.delete(f"/api/admin/users/{student_id}", headers=admin_headers)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/api/llm/ws/chat?token={student_token}"):
            pass
    assert exc_info.value.code == 4401
