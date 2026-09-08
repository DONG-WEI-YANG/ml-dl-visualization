from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.dependencies import get_current_user
from app.analytics.tracker import record_event
from app.analytics.models import LearningEvent
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
async def submit_quiz(submission: QuizSubmission, credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))):
    """Submit quiz answers and get graded results."""
    user = get_current_user(credentials) if credentials else None
    try:
        result = grade_quiz(submission.week, submission.answers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if user:
        record_event(LearningEvent(student_id=str(user['id']), week=submission.week,
                                   event_type='quiz', score=result['percentage']))
    return {"week": submission.week, **result}
