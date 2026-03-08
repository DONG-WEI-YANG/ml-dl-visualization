"""Integration tests for authentication endpoints."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _admin_token() -> str:
    """Login as default admin and return token."""
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_login_success():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_login_wrong_password():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_login_nonexistent_user():
    resp = client.post("/api/auth/login", json={"username": "nobody", "password": "xxx"})
    assert resp.status_code == 401


def test_me_with_token():
    token = _admin_token()
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "admin"


def test_me_without_token():
    resp = client.get("/api/auth/me")
    # HTTPBearer returns 401 for missing credentials
    assert resp.status_code in (401, 403)


def test_me_invalid_token():
    resp = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401


def test_register_and_login_new_user():
    token = _admin_token()
    resp = client.post(
        "/api/auth/register",
        json={"username": "teststudent", "password": "test123", "role": "student"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "teststudent"

    # Login as new user
    resp = client.post("/api/auth/login", json={"username": "teststudent", "password": "test123"})
    assert resp.status_code == 200


def test_register_duplicate_username():
    token = _admin_token()
    # admin already exists
    resp = client.post(
        "/api/auth/register",
        json={"username": "admin", "password": "x", "role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 409


def test_register_without_admin():
    resp = client.post(
        "/api/auth/register",
        json={"username": "hacker", "password": "x", "role": "admin"},
    )
    # HTTPBearer returns 401 for missing credentials
    assert resp.status_code in (401, 403)
