"""Self-service password change."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _make_user(username, password="initpass1", must_change=1):
    admin = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()
    client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "role": "student"},
        headers={"Authorization": f"Bearer {admin['access_token']}"},
    )
    conn = get_db()
    conn.execute("UPDATE users SET must_change_password = ? WHERE username = ?", (must_change, username))
    conn.commit()
    conn.close()
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["access_token"]


def test_change_password_success_clears_flag_and_audits():
    token = _make_user("pwuser1")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "initpass1", "new_password": "newpassword9"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    # old password no longer works, new one does
    assert client.post("/api/auth/login", json={"username": "pwuser1", "password": "initpass1"}).status_code == 401
    login = client.post("/api/auth/login", json={"username": "pwuser1", "password": "newpassword9"})
    assert login.status_code == 200
    assert login.json()["user"]["must_change_password"] is False
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.password_change' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None


def test_change_password_wrong_old():
    token = _make_user("pwuser2")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "WRONG", "new_password": "newpassword9"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_change_password_too_short():
    token = _make_user("pwuser3")
    resp = client.post(
        "/api/auth/change-password",
        json={"old_password": "initpass1", "new_password": "short"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400
