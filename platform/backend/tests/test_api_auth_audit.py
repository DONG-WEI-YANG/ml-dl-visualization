"""Login/logout audit trail + must_change_password exposure."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin_token() -> str:
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


def _last_audit(action):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1", (action,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def test_login_success_audited():
    client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    row = _last_audit("login.success")
    assert row is not None
    assert row["actor_username"] == "admin"
    assert row["ip"] != ""


def test_login_failed_audited_with_attempted_username():
    client.post("/api/auth/login", json={"username": "ghost", "password": "nope"})
    row = _last_audit("login.failed")
    assert row is not None
    assert row["actor_id"] is None
    assert "ghost" in row["detail"]


def test_login_response_contains_must_change_password():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    assert "must_change_password" in resp.json()["user"]


def test_logout_audited():
    token = _admin_token()
    resp = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    row = _last_audit("logout")
    assert row is not None and row["actor_username"] == "admin"
