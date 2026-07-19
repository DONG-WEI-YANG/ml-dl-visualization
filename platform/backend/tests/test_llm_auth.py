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
