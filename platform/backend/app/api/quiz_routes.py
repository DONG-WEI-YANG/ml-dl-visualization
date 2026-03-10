from fastapi import APIRouter
from pydantic import BaseModel
from app.quiz.questions import get_questions_for_week, grade_quiz

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])


@router.get("/week/{week}")
async def quiz_questions(week: int):
    """Get quiz questions for a specific week (no answers included)."""
    questions = get_questions_for_week(week)
    return {"week": week, "questions": questions}


class QuizSubmission(BaseModel):
    week: int
    answers: dict[str, int]  # question_id -> selected option index


@router.post("/submit")
async def submit_quiz(submission: QuizSubmission):
    """Submit quiz answers and get graded results."""
    result = grade_quiz(submission.week, submission.answers)
    return {"week": submission.week, **result}
