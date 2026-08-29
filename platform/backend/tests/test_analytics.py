import pytest
from pydantic import ValidationError

from app.analytics.models import LearningEvent, WeekProgress, StudentAnalytics


def test_learning_event():
    event = LearningEvent(student_id="s001", week=1, event_type="quiz", score=85.0)
    assert event.student_id == "s001"
    assert event.score == 85.0


def test_week_progress():
    wp = WeekProgress(week=1, completed=True, quiz_score=90.0)
    assert wp.completed is True


def test_student_analytics_defaults():
    sa = StudentAnalytics(student_id="s001")
    assert sa.total_weeks_completed == 0
    assert sa.average_score == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("week", 0),
        ("week", 19),
        ("event_type", "made_up"),
        ("score", -0.1),
        ("score", 100.1),
        ("duration_seconds", -1),
    ],
)
def test_learning_event_rejects_values_outside_course_domain(field, value):
    data = {
        "student_id": "s001",
        "week": 1,
        "event_type": "quiz",
        "score": 85.0,
        "duration_seconds": 1,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        LearningEvent(**data)


def test_learning_event_metadata_is_isolated_per_instance():
    first = LearningEvent(student_id="s001", week=1, event_type="quiz")
    second = LearningEvent(student_id="s002", week=1, event_type="quiz")

    first.metadata["attempt"] = 1

    assert second.metadata == {}
