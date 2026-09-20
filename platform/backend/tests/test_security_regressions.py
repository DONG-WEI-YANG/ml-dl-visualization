import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.auth.utils import create_token, hash_password
from app.db import db_connection

client = TestClient(app)


def account(role='student', forced=False):
    name = 'security_' + uuid.uuid4().hex
    with db_connection() as conn:
        user_id = conn.execute(
            'INSERT INTO users(username,password_hash,role,must_change_password) VALUES(?,?,?,?)',
            (name, hash_password('original-password'), role, int(forced)),
        ).lastrowid
    token = create_token(user_id, name, role)
    return user_id, token, {'Authorization': 'Bearer ' + token}


def test_initial_password_only_allows_account_recovery_routes():
    _, token, headers = account('admin', True)
    assert client.get('/api/auth/me', headers=headers).status_code == 200
    assert client.get('/api/admin/users', headers=headers).status_code == 403
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect('/api/llm/ws/chat?token=' + token):
            pass
    assert exc.value.code == 4403


def test_password_change_rotates_token_and_invalidates_other_sessions():
    uid, old, headers = account(forced=True)
    with db_connection() as conn:
        name = conn.execute('SELECT username FROM users WHERE id=?', (uid,)).fetchone()[0]
    other = create_token(uid, name, 'student')
    response = client.post('/api/auth/change-password', headers=headers,
        json={'old_password': 'original-password', 'new_password': 'replacement-password'})
    assert response.status_code == 200
    assert client.get('/api/auth/me', headers=headers).status_code == 401
    assert client.get('/api/auth/me', headers={'Authorization': 'Bearer ' + other}).status_code == 401
    new_token = response.json()['access_token']
    assert new_token != old
    assert client.get('/api/auth/me', headers={'Authorization': 'Bearer ' + new_token}).json()['must_change_password'] is False


def test_logout_revokes_only_current_session():
    uid, token, headers = account()
    with db_connection() as conn:
        name = conn.execute('SELECT username FROM users WHERE id=?', (uid,)).fetchone()[0]
    other = create_token(uid, name, 'student')
    assert client.post('/api/auth/logout', headers=headers).status_code == 200
    assert client.get('/api/auth/me', headers=headers).status_code == 401
    assert client.get('/api/auth/me', headers={'Authorization': 'Bearer ' + other}).status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/llm/ws/chat?token=' + token):
            pass


def test_admin_reset_revokes_previous_token():
    _, _, admin = account('admin')
    uid, _, headers = account()
    assert client.put(f'/api/admin/users/{uid}', headers=admin, json={'password': 'reset-password'}).status_code == 200
    assert client.get('/api/auth/me', headers=headers).status_code == 401


@pytest.mark.parametrize('event_type', ['quiz', 'assignment', 'llm_chat'])
def test_client_cannot_fabricate_server_generated_activity(event_type):
    uid, _, headers = account()
    response = client.post('/api/analytics/events', headers=headers,
        json={'student_id': str(uid), 'week': 1, 'event_type': event_type, 'score': 100})
    assert response.status_code in (403, 422)
    with db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM learning_events WHERE student_id=?', (str(uid),)).fetchone()[0] == 0


def test_visualization_events_cannot_carry_grades_or_client_timestamps():
    uid, _, headers = account()
    payload = {'student_id': str(uid), 'week': 1, 'event_type': 'viz_interaction', 'score': 100}
    assert client.post('/api/analytics/events', headers=headers, json=payload).status_code == 422
    payload.pop('score')
    payload['timestamp'] = '2099-01-01T00:00:00'
    assert client.post('/api/analytics/events', headers=headers, json=payload).status_code == 422


def test_assignment_grading_checks_assignment_and_audits():
    tid, _, teacher = account('teacher')
    sid, _, student = account()
    body = {'student_id': str(sid), 'week': 1, 'score': 82}
    assert client.post('/api/analytics/assignments/grade', headers=student, json=body).status_code == 403
    assert client.post('/api/analytics/assignments/grade', headers=teacher, json=body).status_code == 403
    with db_connection() as conn:
        conn.execute('INSERT INTO teacher_students VALUES(?,?)', (tid, sid))
    result = client.post('/api/analytics/assignments/grade', headers=teacher, json=body)
    assert result.status_code == 200
    assert client.get(f'/api/analytics/students/{sid}', headers=student).json()['average_score'] == 82
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM audit_logs WHERE action='assignment.grade' AND actor_id=?", (tid,)).fetchone()
        assert row is not None


def test_assignment_grade_normalizes_student_identity():
    _, _, admin = account('admin')
    sid, _, student = account()
    result = client.post('/api/analytics/assignments/grade', headers=admin,
                         json={'student_id': '000' + str(sid), 'week': 2, 'score': 75})
    assert result.status_code == 200
    assert result.json()['student_id'] == str(sid)
    assert client.get(f'/api/analytics/students/{sid}', headers=student).json()['average_score'] == 75
    assert client.post('/api/analytics/assignments/grade', headers=admin,
                       json={'student_id': '99999999999999999999', 'week': 2, 'score': 75}).status_code == 422


def test_unchanged_role_does_not_revoke_session_but_actual_change_does():
    _, _, admin = account('admin')
    sid, _, student = account()
    assert client.put(f'/api/admin/users/{sid}', headers=admin,
                      json={'role': 'student', 'is_active': True, 'display_name': 'Updated'}).status_code == 200
    assert client.get('/api/auth/me', headers=student).status_code == 200
    assert client.put(f'/api/admin/users/{sid}', headers=admin, json={'role': 'teacher'}).status_code == 200
    assert client.get('/api/auth/me', headers=student).status_code == 401


def test_open_websocket_rechecks_revocation_before_next_request(monkeypatch):
    async def events(*args, **kwargs):
        yield {'type': 'done'}
    monkeypatch.setattr('app.api.llm_routes._stream_chat_events', events)
    _, token, headers = account()
    with client.websocket_connect('/api/llm/ws/chat?token=' + token) as ws:
        ws.send_json({'messages': []})
        assert ws.receive_json()['type'] == 'done'
        assert client.post('/api/auth/logout', headers=headers).status_code == 200
        ws.send_json({'messages': []})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_production_rejects_default_or_short_jwt_secret():
    from app.config import Settings
    from pydantic import ValidationError
    valid = Settings(_env_file=None, app_env='production', jwt_secret='strong-secret-for-production-0123456789')
    assert valid.app_env == 'production'
    for secret in ('short', 'change-me-in-production-use-a-long-random-string'):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, app_env='production', jwt_secret=secret)


def test_production_admin_initialization_requires_strong_password(tmp_path, monkeypatch):
    from app import db
    from app.config import settings
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'production.db')
    monkeypatch.setattr(settings, 'app_env', 'production')
    monkeypatch.setattr(settings, 'default_admin_password', 'admin123')
    with pytest.raises(RuntimeError, match='DEFAULT_ADMIN_PASSWORD'):
        db.init_db()
    monkeypatch.setattr(settings, 'default_admin_password', 'strong-bootstrap-password')
    db.init_db()
    # A later restart never replaces an existing administrator's credentials.
    monkeypatch.setattr(settings, 'default_admin_password', 'admin123')
    db.init_db()
    from app.auth.utils import verify_password
    with db.db_connection() as conn:
        admin = conn.execute("SELECT * FROM users WHERE username='admin'").fetchone()
    assert verify_password('strong-bootstrap-password', admin['password_hash'])
    assert admin['must_change_password'] == 1
