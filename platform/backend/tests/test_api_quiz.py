"""Integration tests for quiz endpoints."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_quiz_week_1():
    resp = client.get("/api/quiz/week/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["week"] == 1
    assert len(data["questions"]) == 3
    # Answers should not be included
    for q in data["questions"]:
        assert "id" in q
        assert "question" in q
        assert "options" in q
        assert "answer" not in q


def test_get_quiz_week_18():
    resp = client.get("/api/quiz/week/18")
    assert resp.status_code == 200
    assert len(resp.json()["questions"]) == 3


def test_get_quiz_invalid_week():
    resp = client.get("/api/quiz/week/99")
    assert resp.status_code == 200
    assert resp.json()["questions"] == []


def test_submit_quiz():
    # Get questions first
    resp = client.get("/api/quiz/week/1")
    questions = resp.json()["questions"]
    # Submit answers (first option for each)
    answers = {q["id"]: 0 for q in questions}
    resp = client.post("/api/quiz/submit", json={"week": 1, "answers": answers})
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "total" in data
    assert "percentage" in data
    assert data["total"] == 3


def test_submit_quiz_empty_answers():
    resp = client.post("/api/quiz/submit", json={"week": 1, "answers": {}})
    assert resp.status_code == 200
    assert resp.json()["score"] == 0
