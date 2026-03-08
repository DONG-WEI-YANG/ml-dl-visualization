"""Integration tests for analytics endpoints."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_record_event():
    resp = client.post(
        "/api/analytics/events",
        json={
            "student_id": "test-001",
            "week": 1,
            "event_type": "quiz",
            "topic": "python basics",
            "score": 85.0,
            "duration_seconds": 300,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    assert "id" in resp.json()


def test_get_student_analytics():
    # Record a few events first
    for i in range(3):
        client.post(
            "/api/analytics/events",
            json={
                "student_id": "test-analytics",
                "week": i + 1,
                "event_type": "assignment",
                "score": 70 + i * 10,
                "duration_seconds": 600,
            },
        )
    resp = client.get("/api/analytics/students/test-analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == "test-analytics"
    assert data["total_weeks_completed"] >= 1
    assert data["total_time_minutes"] >= 0


def test_class_summary():
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_students" in data
    assert "total_events" in data
    assert "average_score" in data
    assert "popular_llm_topics" in data
