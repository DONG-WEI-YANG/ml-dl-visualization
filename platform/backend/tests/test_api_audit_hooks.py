"""Audit coverage of admin mutations."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _last(action):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1", (action,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def test_register_audited():
    client.post(
        "/api/auth/register",
        json={"username": "hookuser1", "password": "somepass1", "role": "student"},
        headers=_admin(),
    )
    row = _last("user.create")
    assert row is not None and row["actor_username"] == "admin"


def test_update_user_audits_changed_fields_not_values():
    headers = _admin()
    uid = client.post(
        "/api/auth/register",
        json={"username": "hookuser2", "password": "somepass1", "role": "student"},
        headers=headers,
    ).json()["id"]
    client.put(f"/api/admin/users/{uid}", json={"email": "new@x.tw", "password": "resetpass1"},
               headers=headers)
    row = _last("user.update")
    assert row is not None
    assert "email" in row["detail"]
    assert "resetpass1" not in row["detail"], "password value must never be logged"
    # password reset via admin also emits a dedicated event
    assert _last("user.password_reset") is not None


def test_settings_update_audited_with_keys():
    client.put("/api/admin/settings", json={"rag_top_k": "7"}, headers=_admin())
    row = _last("settings.update")
    assert row is not None and "rag_top_k" in row["detail"]


def test_quiz_crud_audited():
    headers = _admin()
    client.post(
        "/api/admin/quiz/questions",
        json={"id": "audit-q1", "week": 1, "question": "Q?", "options": ["a", "b"], "answer": 0},
        headers=headers,
    )
    assert _last("quiz.create") is not None
    client.put("/api/admin/quiz/questions/audit-q1", json={"question": "Q2?"}, headers=headers)
    assert _last("quiz.update") is not None
    client.delete("/api/admin/quiz/questions/audit-q1", headers=headers)
    assert _last("quiz.delete") is not None
