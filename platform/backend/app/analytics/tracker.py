import json
from datetime import datetime
from .models import LearningEvent, StudentAnalytics, WeekProgress
from app.db import db_connection


def record_event(event: LearningEvent) -> int:
    ts = event.timestamp or datetime.now()
    with db_connection() as conn:
        user = conn.execute('SELECT semester FROM users WHERE id = ?', (event.student_id,)).fetchone()
        semester = user['semester'] if user else ''
        cursor = conn.execute(
            """INSERT INTO learning_events
               (student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp, semester)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.student_id, event.week, event.event_type, event.topic,
             event.score, event.duration_seconds, json.dumps(event.metadata),
             ts.isoformat(), semester),
        )
        event_id = cursor.lastrowid
    return event_id


def get_student_analytics(student_id: str, semester: str | None = None) -> StudentAnalytics:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM learning_events WHERE student_id = ?" +
            (" AND semester = ?" if semester is not None else "") + " ORDER BY week, timestamp, id",
            (student_id, semester) if semester is not None else (student_id,),
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


def get_roster(semester=None, teacher_id=None, class_name=None, academic_year=None):
    conditions = ["u.role = 'student'", "u.deleted_at IS NULL"]
    params = []
    if teacher_id is not None:
        conditions.append('EXISTS (SELECT 1 FROM teacher_students ts WHERE ts.student_id = u.id AND ts.teacher_id = ?)')
        params.append(teacher_id)
    for column, value in [('e.semester', semester), ('e.class_name', class_name)]:
        if value is not None:
            conditions.append(column + ' = ?')
            params.append(value)
    if academic_year:
        conditions.append("substr(e.semester, 1, instr(e.semester, '-') - 1) = ?")
        params.append(academic_year)
    # Include users created before enrollment snapshots, as well as never-started students.
    with db_connection() as conn:
        rows = conn.execute(
            "WITH placements AS (SELECT student_id, semester, class_name FROM enrollments UNION "
            "SELECT id, semester, class_name FROM users WHERE role = 'student' AND NOT EXISTS "
            "(SELECT 1 FROM enrollments WHERE student_id = users.id AND semester = users.semester) UNION "
            "SELECT u.id, le.semester, '' FROM learning_events le JOIN users u ON le.student_id = CAST(u.id AS TEXT) "
            "WHERE le.semester != u.semester AND NOT EXISTS "
            "(SELECT 1 FROM enrollments WHERE student_id = u.id AND semester = le.semester)) "
            "SELECT u.id, u.username, u.display_name, u.is_active, e.semester, e.class_name, "
            "COUNT(le.id) AS total_events, MAX(le.timestamp) AS last_activity, "
            "COUNT(DISTINCT CASE WHEN le.event_type = 'quiz' AND le.score IS NOT NULL THEN le.week END) AS quiz_weeks, "
            "COUNT(DISTINCT CASE WHEN le.event_type = 'assignment' AND le.score IS NOT NULL THEN le.week END) AS total_weeks_completed, "
            "AVG(le.score) AS average_score, COALESCE(SUM(le.duration_seconds), 0) / 60 AS total_time_minutes "
            "FROM users u JOIN placements e ON e.student_id = u.id "
            "LEFT JOIN learning_events le ON le.student_id = CAST(u.id AS TEXT) AND le.semester = e.semester "
            "WHERE " + ' AND '.join(conditions) + " GROUP BY u.id, e.semester ORDER BY e.semester DESC, e.class_name, u.username",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_class_summary(semester=None, teacher_id=None, class_name=None, academic_year=None):
    roster = get_roster(semester, teacher_id, class_name, academic_year)
    # Aggregate the same enrollment scope as the roster, without dropping zero-progress users.
    conditions = ["u.role = 'student'", "u.deleted_at IS NULL"]
    params = []
    if teacher_id is not None:
        conditions.append('EXISTS (SELECT 1 FROM teacher_students ts WHERE ts.student_id = u.id AND ts.teacher_id = ?)')
        params.append(teacher_id)
    if semester is not None:
        conditions.append('le.semester = ?')
        params.append(semester)
    if academic_year:
        conditions.append("substr(le.semester, 1, instr(le.semester, '-') - 1) = ?")
        params.append(academic_year)
    if class_name is not None:
        conditions.append("COALESCE(e.class_name, CASE WHEN u.semester = le.semester THEN u.class_name END) = ?")
        params.append(class_name)
    query = (" FROM learning_events le JOIN users u ON le.student_id = CAST(u.id AS TEXT) "
             "LEFT JOIN enrollments e ON e.student_id = u.id AND e.semester = le.semester WHERE " + ' AND '.join(conditions))
    with db_connection() as conn:
        totals = conn.execute('SELECT COUNT(*) AS total, AVG(le.score) AS score' + query, params).fetchone()
        topics = conn.execute("SELECT le.topic, COUNT(*) AS cnt" + query +
            " AND le.event_type = 'llm_chat' AND le.topic != '' GROUP BY le.topic ORDER BY cnt DESC LIMIT 10", params).fetchall()
    return {'total_students': len({r['id'] for r in roster}), 'total_events': totals['total'],
            'average_score': round(totals['score'] or 0, 2),
            'popular_llm_topics': [{'topic': r['topic'], 'count': r['cnt']} for r in topics]}
