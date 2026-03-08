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
