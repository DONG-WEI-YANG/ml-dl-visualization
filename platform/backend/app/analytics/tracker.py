import json
from datetime import datetime
from .models import LearningEvent, StudentAnalytics, WeekProgress
from app.db import db_connection


def record_event(event: LearningEvent) -> int:
    ts = event.timestamp or datetime.now()
    with db_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO learning_events
               (student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.student_id, event.week, event.event_type, event.topic,
             event.score, event.duration_seconds, json.dumps(event.metadata),
             ts.isoformat()),
        )
        event_id = cursor.lastrowid
    return event_id


def get_student_analytics(student_id: str) -> StudentAnalytics:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM learning_events WHERE student_id = ? ORDER BY week, timestamp",
            (student_id,),
        ).fetchall()

    weekly: dict[int, WeekProgress] = {}
    llm_topics: dict[str, int] = {}
    error_patterns: dict[str, int] = {}
    total_time = 0

    for row in rows:
        w = row["week"]
        if w not in weekly:
            weekly[w] = WeekProgress(week=w)

        wp = weekly[w]
        total_time += row["duration_seconds"]
        wp.time_spent_minutes += row["duration_seconds"] // 60

        if row["event_type"] == "quiz" and row["score"] is not None:
            wp.quiz_score = row["score"]
        elif row["event_type"] == "assignment" and row["score"] is not None:
            wp.assignment_score = row["score"]
            wp.completed = True
        elif row["event_type"] == "llm_chat":
            wp.llm_interactions += 1
            topic = row["topic"]
            if topic:
                llm_topics[topic] = llm_topics.get(topic, 0) + 1
            # Extract error patterns from metadata
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                if meta.get("error_type"):
                    etype = meta["error_type"]
                    error_patterns[etype] = error_patterns.get(etype, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

    scores = [wp.assignment_score for wp in weekly.values() if wp.assignment_score is not None]

    return StudentAnalytics(
        student_id=student_id,
        total_weeks_completed=sum(1 for wp in weekly.values() if wp.completed),
        total_time_minutes=total_time // 60,
        average_score=sum(scores) / len(scores) if scores else 0.0,
        weekly_progress=sorted(weekly.values(), key=lambda x: x.week),
        llm_topics=[{"topic": k, "count": v} for k, v in sorted(llm_topics.items(), key=lambda x: -x[1])],
        error_patterns=[{"type": k, "count": v} for k, v in sorted(error_patterns.items(), key=lambda x: -x[1])],
    )


def get_class_summary(semester: str | None = None, teacher_id: int | None = None) -> dict:
    joins = []
    conditions = []
    params: list[object] = []
    if semester:
        joins.append("JOIN users u ON le.student_id = CAST(u.id AS TEXT)")
        conditions.append("u.semester = ?")
        params.append(semester)
    if teacher_id is not None:
        joins.append("JOIN teacher_students ts ON le.student_id = CAST(ts.student_id AS TEXT)")
        conditions.append("ts.teacher_id = ?")
        params.append(teacher_id)
    join_sql = " ".join(joins)
    where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    with db_connection() as conn:
        students = conn.execute(
            f"SELECT DISTINCT le.student_id FROM learning_events le {join_sql}{where_sql}",
            params,
        ).fetchall()
        total_events = conn.execute(
            f"SELECT COUNT(*) as c FROM learning_events le {join_sql}{where_sql}",
            params,
        ).fetchone()["c"]
        score_conditions = [*conditions, "le.score IS NOT NULL"]
        score_where = f" WHERE {' AND '.join(score_conditions)}"
        avg_score = conn.execute(
            f"SELECT AVG(le.score) as avg FROM learning_events le {join_sql}{score_where}",
            params,
        ).fetchone()["avg"]
        topic_conditions = [*conditions, "le.event_type = 'llm_chat'", "le.topic != ''"]
        topic_where = f" WHERE {' AND '.join(topic_conditions)}"
        popular_topics = conn.execute(
            f"SELECT le.topic, COUNT(*) as cnt FROM learning_events le {join_sql}{topic_where} "
            "GROUP BY le.topic ORDER BY cnt DESC LIMIT 10",
            params,
        ).fetchall()

    return {
        "total_students": len(students),
        "total_events": total_events,
        "average_score": round(avg_score, 2) if avg_score else 0,
        "popular_llm_topics": [{"topic": r["topic"], "count": r["cnt"]} for r in popular_topics],
    }
