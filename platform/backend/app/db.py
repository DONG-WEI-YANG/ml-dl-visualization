import logging
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from app.config import settings

DB_PATH = settings.database_path
logger = logging.getLogger(__name__)

# Bookkeeping keys stored in system_settings that are not user-facing config.
INTERNAL_SETTING_KEYS = frozenset({"admin_reset_applied", "rag_curriculum_fingerprint"})


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Open one transaction and always release its SQLite connection."""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            email TEXT DEFAULT '',
            semester TEXT NOT NULL DEFAULT '',
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

        CREATE TABLE IF NOT EXISTS quiz_questions (
            id TEXT PRIMARY KEY,
            week INTEGER NOT NULL,
            question TEXT NOT NULL,
            options TEXT NOT NULL,
            answer INTEGER NOT NULL,
            explanation TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT 'concept',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            actor_id INTEGER,
            actor_username TEXT NOT NULL DEFAULT '',
            actor_role TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            target_type TEXT NOT NULL DEFAULT '',
            target_id TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '{}',
            ip TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor_id);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            jti TEXT PRIMARY KEY,
            expires_at INTEGER NOT NULL
        );
    """)
    # Migration: add semester column if not exists
    try:
        conn.execute("ALTER TABLE users ADD COLUMN semester TEXT NOT NULL DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    # Migration: soft-delete + forced password change columns
    for ddl in (
        "ALTER TABLE users ADD COLUMN deleted_at TEXT",
        "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        else:
            if "must_change_password" in ddl:
                # One-shot upgrade: this branch only runs the moment the
                # column is first added, so an existing deployment's admin
                # row (old password, no forced-change flag) is forced to
                # change its password on next login.
                conn.execute(
                    "UPDATE users SET must_change_password = 1 WHERE username = 'admin'"
                )
                conn.commit()
    # Seed default admin if none exists
    # Additive migration: unknown historical event terms remain unclassified.
    for table, column, definition in (
        ('users', 'class_name', "TEXT NOT NULL DEFAULT ''"),
        # Self-service sign-up: 'pending' accounts cannot log in until approved.
        ('users', 'approval_status', "TEXT NOT NULL DEFAULT 'approved'"),
        ('learning_events', 'semester', "TEXT NOT NULL DEFAULT ''"),
    ):
        columns = {r[1] for r in conn.execute(f'PRAGMA table_info({table})')}
        if column not in columns:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS enrollments (
            student_id INTEGER NOT NULL REFERENCES users(id),
            semester TEXT NOT NULL,
            class_name TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(student_id, semester)
        );
        INSERT OR IGNORE INTO enrollments(student_id, semester, class_name)
        SELECT id, semester, class_name FROM users WHERE role = 'student';
        CREATE INDEX IF NOT EXISTS idx_events_student_term ON learning_events(student_id, semester);
    """)
    existing = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
    if not existing:
        from app.auth.utils import hash_password
        from app.config import settings
        if settings.app_env == 'production' and (
            len(settings.default_admin_password) < 12
            or settings.default_admin_password == 'admin123'
        ):
            conn.close()
            raise RuntimeError('Initial production administrator requires DEFAULT_ADMIN_PASSWORD of at least 12 characters')
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role, must_change_password) "
            "VALUES (?, ?, ?, ?, 1)",
            ("admin", hash_password(settings.default_admin_password), "系統管理員", "admin"),
        )
    _apply_admin_reset(conn)
    # Seed default LLM settings
    defaults = {
        "llm_provider": "local",
        "llm_model": "local-nlp",
        "rag_enabled": "true",
        "rag_top_k": "5",
        "current_semester": "114-2",
    }
    for k, v in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)", (k, v)
        )
    from app.quiz.seed import seed_quiz_questions
    seed_quiz_questions(conn)
    conn.commit()
    conn.close()


_ADMIN_RESET_MARKER = "admin_reset_applied"


def _apply_admin_reset(conn) -> None:
    """Reset the `admin` password from ADMIN_RESET_PASSWORD, once per value.

    The marker stores a salted hash of the applied value, so restarting with
    the secret still set does not undo a password the admin changed later.
    """
    from app.auth.utils import hash_password, verify_password
    from app.config import settings
    # Trim: pasted secrets often carry a trailing newline, and the login page
    # trims what the user types, so an untrimmed value could never match.
    value = settings.admin_reset_password.strip()
    if not value:
        return
    if len(value) < 12:
        logger.warning("ADMIN_RESET_PASSWORD ignored: must be at least 12 characters")
        return
    marker = conn.execute(
        "SELECT value FROM system_settings WHERE key = ?", (_ADMIN_RESET_MARKER,)
    ).fetchone()
    if marker and verify_password(value, marker[0]):
        return
    cursor = conn.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 1, is_active = 1, deleted_at = NULL, "
        "approval_status = 'approved', session_version = session_version + 1, updated_at = datetime('now') "
        "WHERE username = 'admin'",
        (hash_password(value),),
    )
    if cursor.rowcount == 0:
        return
    conn.execute(
        "INSERT INTO system_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (_ADMIN_RESET_MARKER, hash_password(value)),
    )
    conn.execute(
        "INSERT INTO audit_logs (actor_username, actor_role, action, target_type, target_id, detail) "
        "VALUES ('system', 'system', 'user.password_reset', 'user', 'admin', '{\"via\": \"ADMIN_RESET_PASSWORD\"}')"
    )
    logger.warning("Admin password reset from ADMIN_RESET_PASSWORD; remove the secret after logging in")


def sync_enrollment(conn, user_id: int):
    """Snapshot the current placement without overwriting earlier semesters."""
    conn.execute(
        "INSERT INTO enrollments(student_id, semester, class_name) "
        "SELECT id, semester, class_name FROM users WHERE id = ? AND role = 'student' "
        "ON CONFLICT(student_id, semester) DO UPDATE SET class_name = excluded.class_name",
        (user_id,),
    )


def get_setting(key: str, default: str = "") -> str:
    with db_connection() as conn:
        row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO system_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value),
        )


def get_all_settings() -> dict[str, str]:
    with db_connection() as conn:
        rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
    return {r["key"]: r["value"] for r in rows if r["key"] not in INTERNAL_SETTING_KEYS}
