import sqlite3

import pytest

import app.db as db_module


def test_db_connection_commits_and_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "commit.db")

    with db_module.db_connection() as conn:
        conn.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sample VALUES ('saved')")
        closed_connection = conn

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        closed_connection.execute("SELECT 1")

    verify = db_module.get_db()
    try:
        assert verify.execute("SELECT value FROM sample").fetchone()["value"] == "saved"
    finally:
        verify.close()


def test_db_connection_rolls_back_and_closes_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "rollback.db")
    setup = db_module.get_db()
    setup.execute("CREATE TABLE sample (value TEXT NOT NULL)")
    setup.commit()
    setup.close()

    with pytest.raises(RuntimeError, match="boom"):
        with db_module.db_connection() as conn:
            conn.execute("INSERT INTO sample VALUES ('discarded')")
            failed_connection = conn
            raise RuntimeError("boom")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        failed_connection.execute("SELECT 1")

    verify = db_module.get_db()
    try:
        assert verify.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 0
    finally:
        verify.close()
