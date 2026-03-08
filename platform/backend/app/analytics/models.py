from pydantic import BaseModel
from datetime import datetime


class LearningEvent(BaseModel):
    student_id: str
    week: int
    event_type: str  # "quiz" | "assignment" | "llm_chat" | "viz_interaction"
    topic: str = ""
    score: float | None = None
    duration_seconds: int = 0
    metadata: dict = {}
    timestamp: datetime | None = None


class WeekProgress(BaseModel):
    week: int
    completed: bool = False
    quiz_score: float | None = None
    assignment_score: float | None = None
    llm_interactions: int = 0
    time_spent_minutes: int = 0


class StudentAnalytics(BaseModel):
    student_id: str
    total_weeks_completed: int = 0
    total_time_minutes: int = 0
    average_score: float = 0.0
    weekly_progress: list[WeekProgress] = []
    llm_topics: list[dict] = []
    error_patterns: list[dict] = []
