import json
from app.db import get_db


def get_questions_for_week(week: int) -> list[dict]:
    """Fetch quiz questions for a given week from DB (without answers)."""
    db = get_db()
    rows = db.execute(
        "SELECT id, week, question, options, category FROM quiz_questions WHERE week = ? ORDER BY id",
        (week,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "week": row["week"],
            "question": row["question"],
            "options": json.loads(row["options"]),
            "category": row["category"],
        }
        for row in rows
    ]


def get_question_by_id(question_id: str) -> dict | None:
    """Fetch a single question by ID (with answer)."""
    db = get_db()
    row = db.execute("SELECT * FROM quiz_questions WHERE id = ?", (question_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "week": row["week"],
        "question": row["question"],
        "options": json.loads(row["options"]),
        "answer": row["answer"],
        "explanation": row["explanation"],
        "category": row["category"],
    }


def grade_quiz(week: int, answers: dict[str, int]) -> dict:
    """Grade quiz answers against DB questions."""
    db = get_db()
    rows = db.execute(
        "SELECT id, question, options, answer, explanation FROM quiz_questions WHERE week = ?",
        (week,),
    ).fetchall()

    questions = {row["id"]: dict(row) for row in rows}
    results = []
    correct = 0
    total = len(answers)

    for qid, selected in answers.items():
        q = questions.get(qid)
        if not q:
            results.append({"id": qid, "correct": False, "message": "題目不存在"})
            continue
        opts = json.loads(q["options"])
        is_correct = selected == q["answer"]
        if is_correct:
            correct += 1
        results.append({
            "id": qid,
            "correct": is_correct,
            "selected": selected,
            "answer": q["answer"],
            "answer_text": opts[q["answer"]] if q["answer"] < len(opts) else "",
            "explanation": q["explanation"],
        })

    return {
        "week": week,
        "score": correct,
        "total": total,
        "percentage": round(correct / total * 100, 1) if total > 0 else 0,
        "results": results,
    }


# --- Admin CRUD ---

def create_question(data: dict) -> dict:
    """Create a new quiz question."""
    db = get_db()
    db.execute(
        "INSERT INTO quiz_questions (id, week, question, options, answer, explanation, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (data["id"], data["week"], data["question"], json.dumps(data["options"], ensure_ascii=False),
         data["answer"], data.get("explanation", ""), data.get("category", "concept")),
    )
    db.commit()
    return get_question_by_id(data["id"])


def update_question(question_id: str, data: dict) -> dict | None:
    """Update an existing quiz question."""
    db = get_db()
    fields = []
    values = []
    for key in ("week", "question", "answer", "explanation", "category"):
        if key in data:
            fields.append(f"{key} = ?")
            values.append(data[key])
    if "options" in data:
        fields.append("options = ?")
        values.append(json.dumps(data["options"], ensure_ascii=False))
    if not fields:
        return get_question_by_id(question_id)
    values.append(question_id)
    db.execute(f"UPDATE quiz_questions SET {', '.join(fields)} WHERE id = ?", values)
    db.commit()
    return get_question_by_id(question_id)


def delete_question(question_id: str) -> bool:
    """Hard delete a quiz question."""
    db = get_db()
    cursor = db.execute("DELETE FROM quiz_questions WHERE id = ?", (question_id,))
    db.commit()
    return cursor.rowcount > 0


def list_all_questions(week: int | None = None) -> list[dict]:
    """List all questions (with answers) for admin. Optionally filter by week."""
    db = get_db()
    if week is not None:
        rows = db.execute(
            "SELECT * FROM quiz_questions WHERE week = ? ORDER BY id", (week,)
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM quiz_questions ORDER BY week, id").fetchall()
    return [
        {
            "id": row["id"],
            "week": row["week"],
            "question": row["question"],
            "options": json.loads(row["options"]),
            "answer": row["answer"],
            "explanation": row["explanation"],
            "category": row["category"],
        }
        for row in rows
    ]
