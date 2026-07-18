"""Semester batch archiving."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_archive_deactivates_semester_students():
    headers = _admin()
    client.post(
        "/api/admin/users/import",
        json={"semester": "old-sem", "rows": [{"username": "arch1"}, {"username": "arch2"}]},
        headers=headers,
    )
    resp = client.post("/api/admin/semesters/old-sem/archive", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["archived"] == 2
    conn = get_db()
    rows = conn.execute(
        "SELECT is_active FROM users WHERE semester='old-sem' AND role='student'"
    ).fetchall()
    conn.close()
    assert all(r["is_active"] == 0 for r in rows)


def test_archive_empty_semester_returns_zero():
    resp = client.post("/api/admin/semesters/no-such-sem/archive", headers=_admin())
    assert resp.status_code == 200
    assert resp.json()["archived"] == 0


def test_archive_audited():
    _ = _admin()
    client.post("/api/admin/semesters/no-such-sem/archive", headers=_admin())
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='semester.archive' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
