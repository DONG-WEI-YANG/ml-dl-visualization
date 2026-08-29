from fastapi.testclient import TestClient

from app.auth.utils import create_token, hash_password
from app.db import db_connection
from app.main import app


client = TestClient(app)


def _create_user(username: str, role: str) -> tuple[int, dict[str, str]]:
    with db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (username, hash_password("analytics-pass"), username, role),
        )
        user_id = int(cursor.lastrowid)
    token = create_token(user_id, username, role)
    return user_id, {"Authorization": f"Bearer {token}"}


def test_analytics_endpoints_reject_anonymous_requests():
    event = client.post(
        "/api/analytics/events",
        json={"student_id": "1", "week": 1, "event_type": "quiz", "score": 80},
    )
    student = client.get("/api/analytics/students/1")
    summary = client.get("/api/analytics/summary")

    assert event.status_code == 401
    assert student.status_code == 401
    assert summary.status_code == 401


def test_student_event_is_bound_to_authenticated_identity():
    student_id, headers = _create_user("analytics_student_identity", "student")

    response = client.post(
        "/api/analytics/events",
        json={"student_id": "someone-else", "week": 2, "event_type": "quiz", "score": 88},
        headers=headers,
    )

    assert response.status_code == 200
    own = client.get(f"/api/analytics/students/{student_id}", headers=headers)
    assert own.status_code == 200
    assert own.json()["weekly_progress"][0]["quiz_score"] == 88
    spoofed = client.get("/api/analytics/students/someone-else", headers=headers)
    assert spoofed.status_code == 403


def test_student_cannot_read_another_students_analytics():
    first_id, first_headers = _create_user("analytics_student_first", "student")
    second_id, _ = _create_user("analytics_student_second", "student")

    response = client.get(f"/api/analytics/students/{second_id}", headers=first_headers)

    assert first_id != second_id
    assert response.status_code == 403


def test_teacher_can_read_only_assigned_students():
    teacher_id, teacher_headers = _create_user("analytics_teacher", "teacher")
    assigned_id, _ = _create_user("analytics_assigned", "student")
    unrelated_id, _ = _create_user("analytics_unrelated", "student")
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO teacher_students (teacher_id, student_id) VALUES (?, ?)",
            (teacher_id, assigned_id),
        )

    allowed = client.get(f"/api/analytics/students/{assigned_id}", headers=teacher_headers)
    denied = client.get(f"/api/analytics/students/{unrelated_id}", headers=teacher_headers)

    assert allowed.status_code == 200
    assert denied.status_code == 403


def test_only_teacher_or_admin_can_read_class_summary():
    _, student_headers = _create_user("analytics_summary_student", "student")
    _, teacher_headers = _create_user("analytics_summary_teacher", "teacher")
    _, admin_headers = _create_user("analytics_summary_admin", "admin")

    assert client.get("/api/analytics/summary", headers=student_headers).status_code == 403
    assert client.get("/api/analytics/summary", headers=teacher_headers).status_code == 200
    assert client.get("/api/analytics/summary", headers=admin_headers).status_code == 200
