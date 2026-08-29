from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime


class LearningEvent(BaseModel):
    student_id: str = Field(min_length=1, max_length=128)
    week: int = Field(ge=1, le=18)
    event_type: Literal["quiz", "assignment", "llm_chat", "viz_interaction"]
    topic: str = Field(default="", max_length=256)
    score: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    duration_seconds: int = Field(default=0, ge=0, le=86400)
    metadata: dict = Field(default_factory=dict)
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
    weekly_progress: list[WeekProgress] = Field(default_factory=list)
    llm_topics: list[dict] = Field(default_factory=list)
    error_patterns: list[dict] = Field(default_factory=list)
