"""Self-service sign-up with approval, user search, bulk actions, and the
one-shot ADMIN_RESET_PASSWORD recovery path."""
import sqlite3
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import app.db as db_module
from app.config import settings
from app.main import app

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _signup(username, password="studentpass1", **extra):
    body = {"username": username, "password": password, "display_name": "申請者", "class_name": "護理一甲", **extra}
    return client.post("/api/auth/signup", json=body)


# ── Sign-up requests ──

def test_signup_creates_pending_student_that_cannot_login():
    resp = _signup("s9001")
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "student"
    assert data["approval_status"] == "pending"
    assert data["is_active"] is False
    login = client.post("/api/auth/login", json={"username": "s9001", "password": "studentpass1"})
    assert login.status_code == 403
    assert "審核" in login.json()["detail"]


def test_signup_ignores_requested_role():
    resp = _signup("s9002", role="admin")
    assert resp.status_code == 201
    assert resp.json()["role"] == "student"


def test_signup_validation():
    assert _signup("s9003", password="short").status_code == 422
    assert _signup("bad name!").status_code == 422
    assert _signup("s9004").status_code == 201
    assert _signup("s9004").status_code == 409


def test_admin_approves_signup_then_student_can_login():
    uid = _signup("s9010").json()["id"]
    pending = client.get("/api/admin/users?status=pending", headers=_admin()).json()
    assert any(u["id"] == uid for u in pending)
    assert all(u["approval_status"] == "pending" for u in pending)

    resp = client.post(f"/api/admin/users/{uid}/approve", headers=_admin())
    assert resp.status_code == 200
    assert resp.json()["approval_status"] == "approved"
    assert resp.json()["is_active"] is True
    login = client.post("/api/auth/login", json={"username": "s9010", "password": "studentpass1"})
    assert login.status_code == 200
    assert login.json()["user"]["must_change_password"] is False


def test_admin_rejects_signup():
    uid = _signup("s9011").json()["id"]
    resp = client.post(f"/api/admin/users/{uid}/reject", headers=_admin())
    assert resp.status_code == 200
    assert resp.json()["approval_status"] == "rejected"
    login = client.post("/api/auth/login", json={"username": "s9011", "password": "studentpass1"})
    assert login.status_code == 403
    assert "未通過" in login.json()["detail"]


def test_approve_requires_admin():
    uid = _signup("s9012").json()["id"]
    assert client.post(f"/api/admin/users/{uid}/approve").status_code in (401, 403)


def test_approve_only_pending_accounts():
    resp = client.post("/api/admin/users/1/approve", headers=_admin())  # admin row, already approved
    assert resp.status_code == 400


# ── Search ──

def test_user_search_matches_username_name_email_class():
    client.post("/api/auth/register", headers=_admin(), json={
        "username": "findme01", "password": "p" * 12, "display_name": "陳搜尋",
        "email": "seek@x.tw", "class_name": "資管二乙"})
    for q in ("findme", "陳搜", "seek@", "資管二"):
        found = client.get(f"/api/admin/users?q={q}", headers=_admin()).json()
        assert [u["username"] for u in found] == ["findme01"], q


# ── Bulk actions ──

def _make_students(*names):
    ids = []
    for n in names:
        r = client.post("/api/auth/register", headers=_admin(),
                        json={"username": n, "password": "p" * 12, "role": "student"})
        ids.append(r.json()["id"])
    return ids


def test_bulk_deactivate_and_activate():
    ids = _make_students("bulk01", "bulk02")
    r = client.post("/api/admin/users/bulk", headers=_admin(), json={"ids": ids, "action": "deactivate"})
    assert r.status_code == 200 and r.json()["updated"] == 2
    assert client.post("/api/auth/login", json={"username": "bulk01", "password": "p" * 12}).status_code == 403
    r = client.post("/api/admin/users/bulk", headers=_admin(), json={"ids": ids, "action": "activate"})
    assert r.json()["updated"] == 2
    assert client.post("/api/auth/login", json={"username": "bulk01", "password": "p" * 12}).status_code == 200


def test_bulk_reset_password_returns_new_passwords():
    ids = _make_students("bulk03")
    r = client.post("/api/admin/users/bulk", headers=_admin(), json={"ids": ids, "action": "reset_password"})
    pw = r.json()["passwords"][0]
    assert pw["username"] == "bulk03" and len(pw["initial_password"]) >= 12
    login = client.post("/api/auth/login", json={"username": "bulk03", "password": pw["initial_password"]})
    assert login.status_code == 200 and login.json()["user"]["must_change_password"] is True


def test_bulk_set_class():
    ids = _make_students("bulk04", "bulk05")
    r = client.post("/api/admin/users/bulk", headers=_admin(),
                    json={"ids": ids, "action": "set_class", "class_name": "護理三丙"})
    assert r.json()["updated"] == 2
    users = client.get("/api/admin/users?class_name=護理三丙", headers=_admin()).json()
    assert sorted(u["username"] for u in users) == ["bulk04", "bulk05"]


def test_bulk_skips_own_account():
    me = client.get("/api/auth/me", headers=_admin()).json()["id"]
    r = client.post("/api/admin/users/bulk", headers=_admin(), json={"ids": [me], "action": "deactivate"})
    assert r.json()["updated"] == 0
    assert client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).status_code == 200


def test_bulk_rejects_unknown_action():
    r = client.post("/api/admin/users/bulk", headers=_admin(), json={"ids": [1], "action": "explode"})
    assert r.status_code == 422


# ── One-shot admin password reset ──

def _fresh_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setattr(db_module, "DB_PATH", Path(tmp.name))
    db_module.init_db()
    return Path(tmp.name)


def _admin_row(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    conn.close()
    return row


def test_admin_reset_password_applies_once(monkeypatch):
    from app.auth.utils import hash_password, verify_password
    path = _fresh_db(monkeypatch)
    monkeypatch.setattr(settings, "admin_reset_password", "Recovery-Pass-2026")
    db_module.init_db()
    row = _admin_row(path)
    assert verify_password("Recovery-Pass-2026", row["password_hash"])
    assert row["must_change_password"] == 1

    # Admin changes password; a restart with the same secret must not undo it.
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE users SET password_hash = ?, must_change_password = 0 WHERE username = 'admin'",
                 (hash_password("Changed-By-Admin-1"),))
    conn.commit()
    conn.close()
    db_module.init_db()
    assert verify_password("Changed-By-Admin-1", _admin_row(path)["password_hash"])


def test_admin_reset_password_strips_pasted_whitespace(monkeypatch):
    """Secrets pasted into a web form often carry a trailing newline/space; the
    login page trims the password, so the stored value must be trimmed too."""
    from app.auth.utils import verify_password
    path = _fresh_db(monkeypatch)
    monkeypatch.setattr(settings, "admin_reset_password", "  Recovery-Pass-2026\n")
    db_module.init_db()
    assert verify_password("Recovery-Pass-2026", _admin_row(path)["password_hash"])


def test_admin_reset_password_ignores_short_values(monkeypatch):
    from app.auth.utils import verify_password
    path = _fresh_db(monkeypatch)
    before = _admin_row(path)["password_hash"]
    monkeypatch.setattr(settings, "admin_reset_password", "short")
    db_module.init_db()
    assert _admin_row(path)["password_hash"] == before
    assert not verify_password("short", before)
