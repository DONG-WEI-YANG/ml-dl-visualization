"""Integration tests for audit log query/export endpoints."""
from fastapi.testclient import TestClient
from app.main import app
from app.audit import log_audit

client = TestClient(app)


def _admin_token() -> str:
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _seed():
    actor = {"id": 1, "username": "admin", "role": "admin"}
    log_audit("user.create", actor=actor, target_type="user", target_id=7, ip="9.9.9.9")
    log_audit("settings.update", actor=actor, target_type="setting", detail={"keys": ["llm_provider"]})
    log_audit("login.failed", detail={"username": "ghost"}, ip="8.8.8.8")


def test_list_requires_admin():
    resp = client.get("/api/admin/audit-logs")
    assert resp.status_code in (401, 403)


def test_list_returns_paginated_items():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?page=1&page_size=2", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert len(data["items"]) == 2
    # newest first
    assert data["items"][0]["id"] > data["items"][1]["id"]


def test_filter_by_action_prefix():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?action_prefix=login", headers=_auth(token))
    assert resp.status_code == 200
    assert all(item["action"].startswith("login") for item in resp.json()["items"])
    assert resp.json()["total"] >= 1


def test_filter_by_actor():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?actor_id=1", headers=_auth(token))
    assert all(item["actor_id"] == 1 for item in resp.json()["items"])


def test_page_size_capped_at_200():
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs?page_size=999", headers=_auth(token))
    assert resp.status_code == 422  # ge/le validation


def test_export_csv():
    _seed()
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs/export?action_prefix=login", headers=_auth(token))
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    body = resp.content.decode("utf-8-sig")
    assert body.splitlines()[0].startswith("id,timestamp,actor_id")
