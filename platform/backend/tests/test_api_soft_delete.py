"""Soft delete preserves data and hides the user."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_student(username):
    headers = _admin()
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "password": "somepass1", "role": "student"},
        headers=headers,
    )
    return resp.json()["id"]


def test_soft_delete_keeps_row_and_learning_events():
    uid = _create_student("sd_user1")
    conn = get_db()
    conn.execute(
        "INSERT INTO learning_events (student_id, week, event_type, timestamp) "
        "VALUES (?, 1, 'quiz', datetime('now'))",
        (str(uid),),
    )
    conn.commit()
    resp = client.delete(f"/api/admin/users/{uid}", headers=_admin())
    assert resp.status_code == 200
    row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    assert row is not None, "user row must be preserved"
    assert row["deleted_at"] is not None
    assert row["is_active"] == 0
    events = conn.execute(
        "SELECT COUNT(*) AS c FROM learning_events WHERE student_id = ?", (str(uid),)
    ).fetchone()["c"]
    conn.close()
    assert events == 1, "learning events must be preserved"


def test_deleted_user_hidden_and_cannot_login():
    uid = _create_student("sd_user2")
    client.delete(f"/api/admin/users/{uid}", headers=_admin())
    listing = client.get("/api/admin/users", headers=_admin()).json()
    assert all(u["id"] != uid for u in listing)
    assert client.get(f"/api/admin/users/{uid}", headers=_admin()).status_code == 404
    login = client.post("/api/auth/login", json={"username": "sd_user2", "password": "somepass1"})
    assert login.status_code == 401


def test_delete_audited():
    uid = _create_student("sd_user3")
    client.delete(f"/api/admin/users/{uid}", headers=_admin())
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.delete' AND target_id=? "
        "ORDER BY id DESC LIMIT 1", (str(uid),)
    ).fetchone()
    conn.close()
    assert row is not None
