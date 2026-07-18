"""Integration tests for audit log query/export endpoints."""
import csv
import io

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


def test_export_csv_sanitizes_formula_injection():
    """A malicious username reaching the CSV (e.g. via login.failed detail, or
    any other raw column such as ip) must not let Excel execute it as a
    formula: cells starting with =, +, -, @, tab, or CR must be quote-prefixed.
    """
    # A raw column (ip) that carries attacker-controlled content unmodified —
    # the direct exploit path: a formula-triggering value written verbatim.
    log_audit("login.failed", detail={"username": "=cmd|calc"}, ip="=cmd|calc")
    token = _admin_token()
    resp = client.get("/api/admin/audit-logs/export?action_prefix=login", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(body))
    header = next(reader)
    ip_idx = header.index("ip")
    detail_idx = header.index("detail")
    row = next(r for r in reader if r[ip_idx].lstrip("'") == "=cmd|calc")
    assert row[ip_idx] == "'=cmd|calc", f"ip cell not sanitized: {row[ip_idx]!r}"
    # detail is JSON-encoded, so it already starts with "{" — but confirm the
    # sanitizer never breaks that (still no leading formula-trigger char).
    assert row[detail_idx][0] not in ("=", "+", "-", "@", "\t", "\r")
