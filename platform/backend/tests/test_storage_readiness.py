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


_CHUNKS = [{'id': 'w01_lecture_000', 'content': 'old text', 'metadata': {}}]


def _run_startup_ingest(monkeypatch, chunks, stored_fingerprint):
    """Run startup RAG init against a populated index; return ingest calls and saved settings."""
    monkeypatch.setattr(store, 'get_stats', lambda: {'curriculum_chunks': 5})
    monkeypatch.setattr(chunker, 'load_curriculum_chunks', lambda: chunks)
    ingested, saved = [], {}
    monkeypatch.setattr(store, 'ingest_chunks', lambda c: ingested.append(c) or len(c))
    monkeypatch.setattr(db, 'get_setting', lambda k, d='': stored_fingerprint if k == main.RAG_FINGERPRINT_KEY else d)
    monkeypatch.setattr(db, 'set_setting', lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(main.readiness, 'status', 'ready')
    monkeypatch.setattr(main.readiness, 'rag', 'pending')
    asyncio.run(main._initialize_rag_background())
    assert main.readiness.rag == 'ready'
    return ingested, saved


def test_unchanged_curriculum_is_not_reingested(monkeypatch):
    fp = main._curriculum_fingerprint(_CHUNKS)
    ingested, saved = _run_startup_ingest(monkeypatch, _CHUNKS, fp)
    assert ingested == []
    assert saved == {}


def test_changed_curriculum_is_reingested_after_deploy(monkeypatch):
    """Persistent DB keeps old chunks across deploys; edited lectures must be re-indexed."""
    old_fp = main._curriculum_fingerprint(_CHUNKS)
    new_chunks = [{**_CHUNKS[0], 'content': 'corrected text'}]
    ingested, saved = _run_startup_ingest(monkeypatch, new_chunks, old_fp)
    assert ingested == [new_chunks]
    assert saved[main.RAG_FINGERPRINT_KEY] == main._curriculum_fingerprint(new_chunks)
    assert saved[main.RAG_FINGERPRINT_KEY] != old_fp


def test_index_without_fingerprint_is_reingested(monkeypatch):
    """Databases indexed before fingerprinting existed have no stored value."""
    ingested, saved = _run_startup_ingest(monkeypatch, _CHUNKS, '')
    assert ingested == [_CHUNKS]
    assert main.RAG_FINGERPRINT_KEY in saved
