"""Integration tests for analytics endpoints."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _admin_identity() -> tuple[int, dict[str, str]]:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    return data["user"]["id"], {"Authorization": f"Bearer {data['access_token']}"}


def test_record_event():
    student_id, headers = _admin_identity()
    resp = client.post(
        "/api/analytics/events",
        json={
            "student_id": "ignored-client-value",
            "week": 1,
            "event_type": "viz_interaction",
            "topic": "python basics",
            "duration_seconds": 300,
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    assert "id" in resp.json()
    assert resp.json()["student_id"] == str(student_id)


def test_get_student_analytics():
    _, headers = _admin_identity()
    response = client.post('/api/auth/register', headers=headers, json={
        'username': 'analytics_graded_student', 'password': 'test-password', 'role': 'student'})
    assert response.status_code == 200
    student_id = response.json()['id']
    # Record a few events first
    for i in range(3):
        client.post(
            "/api/analytics/assignments/grade",
            json={
                "student_id": str(student_id),
                "week": i + 1,
                "score": 70 + i * 10,
            },
            headers=headers,
        )
    resp = client.get(f"/api/analytics/students/{student_id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == str(student_id)
    assert data["total_weeks_completed"] >= 1
    assert data["total_time_minutes"] >= 0


def test_class_summary():
    _, headers = _admin_identity()
    resp = client.get("/api/analytics/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_students" in data
    assert "total_events" in data
    assert "average_score" in data
    assert "popular_llm_topics" in data
