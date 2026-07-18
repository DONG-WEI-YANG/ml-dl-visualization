"""Tests for the audit logging helper and schema."""
from app.audit import log_audit
from app.db import get_db


def _fetch_all_audit():
    conn = get_db()
    rows = conn.execute("SELECT * FROM audit_logs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_audit_table_exists_with_indexes():
    conn = get_db()
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    indexes = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    conn.close()
    assert "audit_logs" in tables
    assert {"idx_audit_timestamp", "idx_audit_actor", "idx_audit_action"} <= indexes


def test_users_has_new_columns():
    conn = get_db()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    conn.close()
    assert "deleted_at" in cols
    assert "must_change_password" in cols


def test_log_audit_writes_row():
    actor = {"id": 1, "username": "admin", "role": "admin"}
    log_audit("user.update", actor=actor, target_type="user", target_id=42,
              detail={"fields": ["email"]}, ip="1.2.3.4")
    rows = [r for r in _fetch_all_audit() if r["action"] == "user.update"]
    assert rows, "audit row not written"
    row = rows[-1]
    assert row["actor_id"] == 1
    assert row["actor_username"] == "admin"
    assert row["target_type"] == "user"
    assert row["target_id"] == "42"
    assert '"email"' in row["detail"]
    assert row["ip"] == "1.2.3.4"


def test_log_audit_without_actor():
    log_audit("login.failed", detail={"username": "ghost"}, ip="5.6.7.8")
    row = [r for r in _fetch_all_audit() if r["action"] == "login.failed"][-1]
    assert row["actor_id"] is None
    assert row["actor_username"] == ""


def test_log_audit_never_raises(monkeypatch):
    import app.audit as audit_mod

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(audit_mod, "get_db", boom)
    # Must not raise
    log_audit("user.create", detail={})
