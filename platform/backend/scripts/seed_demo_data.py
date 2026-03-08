"""Seed demo data for demonstrations and acceptance testing.

Usage:
    cd platform/backend
    python scripts/seed_demo_data.py
"""
import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import init_db, get_db
from app.auth.utils import hash_password

DEMO_STUDENTS = [
    ("student01", "張小明"),
    ("student02", "李美華"),
    ("student03", "王大偉"),
    ("student04", "陳佳琪"),
    ("student05", "林志豪"),
]

DEMO_TEACHER = ("teacher01", "劉教授")

TOPICS = [
    "梯度下降", "決策邊界", "過擬合", "特徵工程", "CNN",
    "RNN", "注意力機制", "損失函數", "正則化", "學習率",
    "資料分割", "交叉驗證", "激活函數", "反向傳播", "批次正規化",
]

ERROR_TYPES = [
    "維度不匹配", "學習率過大", "過擬合", "欠擬合",
    "梯度消失", "資料洩漏", "類別不平衡",
]

EVENT_TYPES = ["quiz", "assignment", "llm_chat", "visualization"]


def seed_users(conn):
    """Create demo teacher and student accounts."""
    # Teacher
    conn.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
        (DEMO_TEACHER[0], hash_password("demo123"), DEMO_TEACHER[1], "teacher"),
    )
    teacher = conn.execute("SELECT id FROM users WHERE username = ?", (DEMO_TEACHER[0],)).fetchone()

    # Students
    for uname, dname in DEMO_STUDENTS:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (uname, hash_password("demo123"), dname, "student"),
        )
        student = conn.execute("SELECT id FROM users WHERE username = ?", (uname,)).fetchone()
        if teacher and student:
            conn.execute(
                "INSERT OR IGNORE INTO teacher_students (teacher_id, student_id) VALUES (?, ?)",
                (teacher["id"], student["id"]),
            )

    conn.commit()
    print(f"  Created {len(DEMO_STUDENTS)} students + 1 teacher (password: demo123)")


def seed_learning_events(conn):
    """Generate realistic learning event data for 12 weeks of progress."""
    base_date = datetime.now() - timedelta(weeks=12)
    event_count = 0

    for uname, dname in DEMO_STUDENTS:
        # Each student progresses through 8-12 weeks
        weeks_done = random.randint(8, 12)
        base_score = random.uniform(60, 85)

        for week in range(1, weeks_done + 1):
            week_date = base_date + timedelta(weeks=week - 1)

            # Quiz event
            quiz_score = min(100, max(30, base_score + random.gauss(0, 10)))
            conn.execute(
                "INSERT INTO learning_events (student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (uname, week, "quiz", TOPICS[week % len(TOPICS)],
                 round(quiz_score, 1), random.randint(180, 600), "{}", week_date.isoformat()),
            )

            # Assignment event
            assign_score = min(100, max(40, base_score + random.gauss(5, 8)))
            conn.execute(
                "INSERT INTO learning_events (student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (uname, week, "assignment", TOPICS[week % len(TOPICS)],
                 round(assign_score, 1), random.randint(1200, 3600), "{}", week_date.isoformat()),
            )

            # LLM chat events (2-5 per week)
            for _ in range(random.randint(2, 5)):
                topic = random.choice(TOPICS)
                meta = {}
                if random.random() < 0.3:
                    meta["error_type"] = random.choice(ERROR_TYPES)
                conn.execute(
                    "INSERT INTO learning_events (student_id, week, event_type, topic, score, duration_seconds, metadata, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (uname, week, "llm_chat", topic,
                     None, random.randint(60, 300), json.dumps(meta), week_date.isoformat()),
                )
                event_count += 1

            event_count += 2  # quiz + assignment

        # Gradual improvement
        base_score += random.uniform(0, 3)

    conn.commit()
    print(f"  Generated {event_count} learning events across {len(DEMO_STUDENTS)} students")


def main():
    print("Seeding demo data...")
    init_db()
    conn = get_db()
    seed_users(conn)
    seed_learning_events(conn)
    conn.close()
    print("Done! Demo accounts (password: demo123):")
    print(f"  Teacher: {DEMO_TEACHER[0]}")
    for uname, dname in DEMO_STUDENTS:
        print(f"  Student: {uname} ({dname})")


if __name__ == "__main__":
    main()
