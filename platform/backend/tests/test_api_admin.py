"""Integration tests for admin endpoints."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _admin_token() -> str:
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


def test_list_users_admin():
    token = _admin_token()
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


def test_list_users_unauthorized():
    resp = client.get("/api/admin/users")
    assert resp.status_code in (401, 403)


def test_get_settings():
    token = _admin_token()
    resp = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    assert "available_providers" in data


def test_update_settings():
    token = _admin_token()
    resp = client.put(
        "/api/admin/settings",
        json={"llm_provider": "anthropic"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


def test_update_settings_invalid_key():
    token = _admin_token()
    resp = client.put(
        "/api/admin/settings",
        json={"invalid_key": "value"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400


def test_list_teachers():
    token = _admin_token()
    resp = client.get("/api/admin/teachers", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
