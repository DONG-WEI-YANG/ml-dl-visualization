"""One-shot migration upgrade: existing deployments must force the admin
account to change its password the first time the new column is added,
since the pre-existing row keeps its old (possibly default) password."""
import sqlite3
import tempfile
from pathlib import Path

import app.db as db_module
from app.auth.utils import hash_password


def _make_old_schema_db(path: Path):
    """Build a users table as it existed before this feature (no deleted_at,
    no must_change_password), with an admin row already present."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            email TEXT DEFAULT '',
            role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "INSERT INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
        ("admin", hash_password("admin123"), "系統管理員", "admin"),
    )
    conn.commit()
    conn.close()


def test_migration_forces_existing_admin_to_change_password(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    old_db_path = Path(tmp.name)
    _make_old_schema_db(old_db_path)

    monkeypatch.setattr(db_module, "DB_PATH", old_db_path)
    db_module.init_db()

    conn = sqlite3.connect(str(old_db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT must_change_password FROM users WHERE username = 'admin'"
    ).fetchone()
    conn.close()
    assert row["must_change_password"] == 1


def test_migration_is_noop_when_column_already_exists(monkeypatch):
    """Running init_db() again (column already present) must not clobber an
    admin who has already changed their password."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    _make_old_schema_db(db_path)

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.init_db()  # first run adds the column + forces the flag

    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE users SET must_change_password = 0 WHERE username = 'admin'")
    conn.commit()
    conn.close()

    db_module.init_db()  # second run: column already exists, must be a no-op

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT must_change_password FROM users WHERE username = 'admin'"
    ).fetchone()
    conn.close()
    assert row["must_change_password"] == 0
