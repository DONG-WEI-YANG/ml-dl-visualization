"""Integration tests for quiz endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Helper: check if quiz data has been seeded
_quiz_seeded = None

def _has_quiz_data() -> bool:
    global _quiz_seeded
    if _quiz_seeded is None:
        resp = client.get("/api/quiz/week/1")
        _quiz_seeded = len(resp.json().get("questions", [])) > 0
    return _quiz_seeded

needs_quiz_data = pytest.mark.skipif(
    "not _has_quiz_data()",
    reason="Quiz DB not seeded (CI uses empty DB)",
)


@needs_quiz_data
def test_get_quiz_week_1():
    resp = client.get("/api/quiz/week/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["week"] == 1
    assert len(data["questions"]) == 10
    # Answers and explanations should not be included; category should be
    for q in data["questions"]:
        assert "id" in q
        assert "question" in q
        assert "options" in q
        assert "category" in q
        assert "answer" not in q
        assert "explanation" not in q


@needs_quiz_data
def test_get_quiz_week_18():
    resp = client.get("/api/quiz/week/18")
    assert resp.status_code == 200
    assert len(resp.json()["questions"]) == 10


def test_get_quiz_invalid_week():
    resp = client.get("/api/quiz/week/99")
    assert resp.status_code == 200
    assert resp.json()["questions"] == []


@needs_quiz_data
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
    assert data["total"] == 10


def test_submit_quiz_empty_answers():
    resp = client.post("/api/quiz/submit", json={"week": 1, "answers": {}})
    assert resp.status_code == 200
    assert resp.json()["score"] == 0
