import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            email TEXT DEFAULT '',
            role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS learning_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            week INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            topic TEXT DEFAULT '',
            score REAL,
            duration_seconds INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS teacher_students (
            teacher_id INTEGER NOT NULL REFERENCES users(id),
            student_id INTEGER NOT NULL REFERENCES users(id),
            PRIMARY KEY (teacher_id, student_id)
        );

        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    # Seed default admin if none exists
    existing = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
    if not existing:
        from app.auth.utils import hash_password
        from app.config import settings
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            ("admin", hash_password(settings.default_admin_password), "系統管理員", "admin"),
        )
    # Seed default LLM settings
    defaults = {
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-20250514",
        "rag_enabled": "true",
        "rag_top_k": "5",
    }
    for k, v in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)", (k, v)
        )
    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "") -> str:
    conn = get_db()
    row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO system_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value),
    )
    conn.commit()
    conn.close()


def get_all_settings() -> dict[str, str]:
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}
