import asyncio
import sqlite3

from app import db
from app import main
from app.rag import store, chunker


def test_rag_uses_shared_database_path(tmp_path, monkeypatch):
    target = tmp_path / 'shared.db'
    monkeypatch.setattr(db, 'DB_PATH', target)
    store.init_rag_tables()
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='rag_chunks'").fetchone()


def test_rag_missing_curriculum_is_unavailable(monkeypatch):
    monkeypatch.setattr(store, 'get_stats', lambda: {'curriculum_chunks': 0})
    monkeypatch.setattr(chunker, 'load_curriculum_chunks', lambda: [])
    monkeypatch.setattr(main.readiness, 'status', 'ready')
    monkeypatch.setattr(main.readiness, 'rag', 'pending')
    asyncio.run(main._initialize_rag_background())
    assert main.readiness.rag == 'unavailable'
    assert main.readiness.status == 'degraded'


def test_rag_ingestion_failure_is_error(monkeypatch):
    monkeypatch.setattr(store, 'get_stats', lambda: {'curriculum_chunks': 0})
    monkeypatch.setattr(chunker, 'load_curriculum_chunks', lambda: [{}])
    def fail(_):
        raise RuntimeError('index failure')
    monkeypatch.setattr(store, 'ingest_chunks', fail)
    monkeypatch.setattr(main.readiness, 'status', 'ready')
    monkeypatch.setattr(main.readiness, 'rag', 'pending')
    asyncio.run(main._initialize_rag_background())
    assert main.readiness.rag == 'error'
    assert main.readiness.status == 'degraded'


def test_backup_and_restore_roundtrip_preserves_rows(tmp_path):
    from scripts.backup_database import copy_database
    source, backup, restored = [tmp_path / name for name in ('source.db', 'backup.db', 'restored.db')]
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)')
        conn.execute("INSERT INTO sample VALUES (1, 'retained')")
    copy_database(source, backup)
    copy_database(backup, restored)
    with sqlite3.connect(restored) as conn:
        assert conn.execute('SELECT * FROM sample').fetchall() == [(1, 'retained')]
    import pytest
    with pytest.raises(FileExistsError):
        copy_database(source, restored)


def test_existing_curriculum_is_ready_without_reingestion(monkeypatch):
    monkeypatch.setattr(store, 'get_stats', lambda: {'curriculum_chunks': 5})
    def unexpected():
        raise AssertionError('Existing curriculum should not be reloaded')
    monkeypatch.setattr(chunker, 'load_curriculum_chunks', unexpected)
    monkeypatch.setattr(main.readiness, 'status', 'ready')
    monkeypatch.setattr(main.readiness, 'rag', 'pending')
    asyncio.run(main._initialize_rag_background())
    assert main.readiness.rag == 'ready'
    assert main.readiness.status == 'ready'
