import pytest
from fastapi.testclient import TestClient

from app.db import db_connection
from app.main import app


client = TestClient(app)
QUESTION_IDS = ("integrity-q1", "integrity-q2")


@pytest.fixture(autouse=True)
def integrity_questions():
    with db_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO quiz_questions "
            "(id, week, question, options, answer, explanation, category) "
            "VALUES (?, 17, ?, '[\"A\", \"B\"]', ?, ?, 'concept')",
            [
                (QUESTION_IDS[0], "第一題", 0, "第一題解說"),
                (QUESTION_IDS[1], "第二題", 1, "第二題解說"),
            ],
        )
    yield
    with db_connection() as conn:
        conn.executemany("DELETE FROM quiz_questions WHERE id = ?", [(qid,) for qid in QUESTION_IDS])


def test_partial_submission_counts_every_stored_question():
    response = client.post(
        "/api/quiz/submit",
        json={"week": 17, "answers": {QUESTION_IDS[0]: 0}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["score"] == 1
    assert data["total"] == 2
    assert data["percentage"] == 50.0
    assert data["results"] == [
        {
            "id": QUESTION_IDS[0],
            "correct": True,
            "user_answer": 0,
            "correct_answer": 0,
            "answer_text": "A",
            "explanation": "第一題解說",
        },
        {
            "id": QUESTION_IDS[1],
            "correct": False,
            "user_answer": None,
            "correct_answer": 1,
            "answer_text": "B",
            "explanation": "第二題解說",
        },
    ]


def test_unknown_question_ids_do_not_change_denominator():
    response = client.post(
        "/api/quiz/submit",
        json={"week": 17, "answers": {QUESTION_IDS[0]: 0, "not-in-week": 0}},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["ignored_question_ids"] == ["not-in-week"]


@pytest.mark.parametrize("selected", [-1, 2])
def test_submission_rejects_option_index_outside_question_options(selected):
    response = client.post(
        "/api/quiz/submit",
        json={"week": 17, "answers": {QUESTION_IDS[0]: selected}},
    )

    assert response.status_code == 422
    assert "選項索引" in response.json()["detail"]


def test_empty_week_has_zero_authoritative_total():
    response = client.post(
        "/api/quiz/submit",
        json={"week": 16, "answers": {"unknown": 0}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "week": 16,
        "score": 0,
        "total": 0,
        "percentage": 0,
        "results": [],
        "ignored_question_ids": ["unknown"],
    }
