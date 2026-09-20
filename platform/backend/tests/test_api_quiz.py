"""Integration tests for quiz endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def seeded_quiz_db(tmp_path, monkeypatch):
    """Clone the isolated test DB and seed real curriculum questions per test."""
    import sqlite3
    import app.db as db_module
    from app.quiz.seed import seed_quiz_questions
    path = tmp_path / 'quiz.db'
    source = db_module.get_db()
    target = sqlite3.connect(path)
    source.backup(target)
    source.close()
    target.execute('DELETE FROM quiz_questions')
    seed_quiz_questions(target)
    target.commit()
    target.close()
    monkeypatch.setattr(db_module, 'DB_PATH', path)


def test_get_quiz_week_1():
    resp = client.get("/api/quiz/week/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["week"] == 1
    assert len(data["questions"]) == 3
    # Answers and explanations should not be included; category should be
    for q in data["questions"]:
        assert "id" in q
        assert "question" in q
        assert "options" in q
        assert "category" in q
        assert "answer" not in q
        assert "explanation" not in q


@pytest.mark.parametrize("week", range(2, 19))
def test_get_quiz_all_other_weeks(week):
    resp = client.get(f"/api/quiz/week/{week}")
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


def _admin_headers():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_admin_create_question_missing_fields_is_chinese():
    resp = client.post("/api/admin/quiz/questions", json={"id": "err-q1"}, headers=_admin_headers())
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "缺少必要欄位" in detail
    assert "options" in detail and "{" not in detail


def test_admin_create_question_too_few_options_is_chinese():
    resp = client.post(
        "/api/admin/quiz/questions",
        json={"id": "err-q2", "week": 1, "question": "Q?", "options": ["a"], "answer": 0},
        headers=_admin_headers(),
    )
    assert resp.status_code == 400
    assert "選項" in resp.json()["detail"]


def test_admin_create_question_bad_answer_index_is_chinese():
    resp = client.post(
        "/api/admin/quiz/questions",
        json={"id": "err-q3", "week": 1, "question": "Q?", "options": ["a", "b"], "answer": 5},
        headers=_admin_headers(),
    )
    assert resp.status_code == 400
    assert "答案" in resp.json()["detail"]


def test_admin_create_question_duplicate_id_is_chinese():
    headers = _admin_headers()
    resp = client.post(
        "/api/admin/quiz/questions",
        json={"id": "err-q4", "week": 1, "question": "Q?", "options": ["a", "b"], "answer": 0},
        headers=headers,
    )
    assert resp.status_code == 200
    resp = client.post(
        "/api/admin/quiz/questions",
        json={"id": "err-q4", "week": 1, "question": "Q?", "options": ["a", "b"], "answer": 0},
        headers=headers,
    )
    assert resp.status_code == 400
    assert "已存在" in resp.json()["detail"]


def test_admin_update_question_not_found_is_chinese():
    resp = client.put(
        "/api/admin/quiz/questions/does-not-exist", json={"question": "x"}, headers=_admin_headers()
    )
    assert resp.status_code == 404
    assert "不存在" in resp.json()["detail"]


def test_admin_delete_question_not_found_is_chinese():
    resp = client.delete("/api/admin/quiz/questions/does-not-exist", headers=_admin_headers())
    assert resp.status_code == 404
    assert "不存在" in resp.json()["detail"]

@pytest.mark.parametrize('week', range(1, 19))
def test_seed_answers_grade_to_full_score(week):
    from app.quiz.seed import load_seed
    questions = [q for q in load_seed()['questions'] if q['week'] == week]
    response = client.post('/api/quiz/submit', json={
        'week': week, 'answers': {q['id']: q['answer'] for q in questions},
    })
    assert response.status_code == 200
    result = response.json()
    assert result['total'] == len(questions) == 3
    assert result['score'] == 3
    assert result['percentage'] == 100
    assert all(row['correct'] for row in result['results'])
