import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from app.main import app
from app.auth.utils import create_token
from app.db import db_connection, init_db

client = TestClient(app)


def admin_headers():
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    return {"Authorization": "Bearer " + create_token(row['id'], row['username'], row['role'])}


def create(role='student', semester='115-1', class_name='護理一甲'):
    response = client.post('/api/auth/register', headers=admin_headers(), json={
        'username': uuid.uuid4().hex, 'password': 'test-pass-123', 'role': role,
        'semester': semester, 'class_name': class_name,
    })
    assert response.status_code == 200, response.text
    # These enrollment scenarios require a fully onboarded account.
    login = client.post('/api/auth/login', json={'username': response.json()['username'], 'password': 'test-pass-123'})
    changed = client.post('/api/auth/change-password', headers={'Authorization': 'Bearer ' + login.json()['access_token']},
                          json={'old_password': 'test-pass-123', 'new_password': 'onboarded-pass-456'})
    assert changed.status_code == 200
    return response.json()


def test_class_filter_and_zero_progress_roster():
    student = create(class_name='新班' + uuid.uuid4().hex)
    teacher = create(role='teacher')
    client.post(f"/api/admin/teachers/{teacher['id']}/students/{student['id']}", headers=admin_headers())
    headers = {'Authorization': 'Bearer ' + create_token(teacher['id'], teacher['username'], 'teacher')}
    response = client.get('/api/analytics/roster', params={'semester': '115-1'}, headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 1
    row = response.json()[0]
    assert row['class_name'] == student['class_name']
    assert row['total_events'] == 0
    assert row['total_weeks_completed'] == 0
    summary = client.get('/api/analytics/summary', headers=headers).json()
    assert summary['total_students'] == 1
    assert client.get('/api/analytics/roster', headers={'Authorization': 'Bearer ' + create_token(student['id'], student['username'], 'student')}).status_code == 403


def test_retake_preserves_identity_history_and_separates_terms():
    student = create()
    headers = {'Authorization': 'Bearer ' + create_token(student['id'], student['username'], 'student')}
    event = {'student_id': str(student['id']), 'week': 1, 'event_type': 'assignment', 'score': 80, 'semester': 'spoofed'}
    assert client.post('/api/analytics/assignments/grade', json=event, headers=admin_headers()).status_code == 200
    client.put(f"/api/admin/users/{student['id']}", json={'is_active': False}, headers=admin_headers())
    result = client.post('/api/admin/users/import', json={'semester': '115-2', 'class_name': '重修班', 'rows': [{'username': student['username']}]}, headers=admin_headers())
    assert result.status_code == 200
    assert len(result.json()['restored']) == 1
    current = client.get(f"/api/admin/users/{student['id']}", headers=admin_headers()).json()
    assert current['is_active'] is True
    assert current['class_name'] == '重修班'
    old = client.get(f"/api/analytics/students/{student['id']}?semester=115-1", headers=admin_headers()).json()
    new = client.get(f"/api/analytics/students/{student['id']}?semester=115-2", headers=admin_headers()).json()
    assert old['total_weeks_completed'] == 1
    assert new['total_weeks_completed'] == 0
    roster = client.get('/api/analytics/roster?semester=115-1&class_name=護理一甲', headers=admin_headers()).json()
    assert student['id'] in [r['id'] for r in roster]
    init_db()
    assert client.get(f"/api/analytics/students/{student['id']}?semester=115-1", headers=admin_headers()).json() == old


def test_import_never_restores_teacher_as_student():
    teacher = create(role='teacher')
    client.put(f"/api/admin/users/{teacher['id']}", json={'is_active': False}, headers=admin_headers())
    result = client.post('/api/admin/users/import', headers=admin_headers(), json={'rows': [{'username': teacher['username']}]})
    assert result.status_code == 200
    assert len(result.json()['skipped']) == 1
    assert result.json()['restored'] == []


def test_quiz_submission_records_authenticated_score():
    student = create()
    headers = {'Authorization': 'Bearer ' + create_token(student['id'], student['username'], 'student')}
    result = client.post('/api/quiz/submit', json={'week': 1, 'answers': {}}, headers=headers)
    assert result.status_code == 200
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM learning_events WHERE student_id = ? AND event_type = 'quiz'", (str(student['id']),)).fetchone()
    assert row is not None
    assert row['semester'] == '115-1'
    assert row['score'] == result.json()['percentage']
    roster = client.get('/api/analytics/roster?semester=115-1', headers=admin_headers()).json()
    assert next(r for r in roster if r['id'] == student['id'])['quiz_weeks'] == 1


def test_chat_records_authenticated_activity(monkeypatch):
    from app.api import llm_routes
    student = create()
    headers = {'Authorization': 'Bearer ' + create_token(student['id'], student['username'], 'student')}
    monkeypatch.setattr(llm_routes, '_get_provider', lambda: object())
    monkeypatch.setattr(llm_routes, 'AITutor', lambda *a, **kw: SimpleNamespace(ask=AsyncMock(return_value=SimpleNamespace(content='answer', model='test'))))
    response = client.post('/api/llm/chat', headers=headers, json={'week': 2, 'topic': 'regression', 'messages': [{'role': 'user', 'content': 'help'}]})
    assert response.status_code == 200
    with db_connection() as conn:
        row = conn.execute("SELECT semester, topic FROM learning_events WHERE student_id = ? AND event_type = 'llm_chat'", (str(student['id']),)).fetchone()
    assert row is not None
    assert tuple(row) == ('115-1', 'regression')


def test_legacy_migration_is_additive_and_repeatable(tmp_path, monkeypatch):
    import sqlite3
    import app.db as database
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, display_name TEXT DEFAULT '', email TEXT DEFAULT '', role TEXT, is_active INTEGER DEFAULT 1, created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '');
            INSERT INTO users(id, username, password_hash, role) VALUES (44, 'legacy-student', 'unchanged-hash', 'student');
            CREATE TABLE learning_events(id INTEGER PRIMARY KEY, student_id TEXT, week INTEGER, event_type TEXT, topic TEXT DEFAULT '', score REAL, duration_seconds INTEGER DEFAULT 0, metadata TEXT DEFAULT '{}', timestamp TEXT);
            INSERT INTO learning_events VALUES (1, '44', 1, 'quiz', '', 80, 60, '{}', '2025-01-01');
        """)
    monkeypatch.setattr(database, 'DB_PATH', path)
    database.init_db()
    database.init_db()
    with database.db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM learning_events').fetchone()[0] == 1
        assert conn.execute('SELECT semester FROM learning_events').fetchone()[0] == ''
        assert conn.execute("SELECT password_hash FROM users WHERE id = 44").fetchone()[0] == 'unchanged-hash'
        assert conn.execute('SELECT COUNT(*) FROM enrollments WHERE student_id = 44').fetchone()[0] == 1


def test_unclassified_events_remain_visible_for_already_classified_users():
    from app.analytics.tracker import get_roster
    student = create()
    with db_connection() as conn:
        conn.execute("INSERT INTO learning_events(student_id, week, event_type, score, timestamp) VALUES (?, 1, 'quiz', 55, '2025-01-01')", (str(student['id']),))
    rows = [r for r in get_roster() if r['id'] == student['id']]
    assert any(r['semester'] == '' and r['total_events'] == 1 for r in rows)
