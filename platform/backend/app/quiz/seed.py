"""Versioned curriculum quiz import; no users, network, commits or overwrites."""
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

SEED_VERSION = '2026-09-20.1'
MIN_QUESTIONS_PER_WEEK = 3
SEED_PATH = Path(__file__).with_name('seed_v1.json')


def validate_seed(bundle: dict) -> None:
    if not isinstance(bundle, dict) or bundle.get('version') != SEED_VERSION:
        raise ValueError('Unsupported quiz seed version')
    questions = bundle.get('questions')
    if not isinstance(questions, list):
        raise ValueError('Quiz seed questions must be a list')
    seen = set()
    counts = Counter()
    for q in questions:
        if not isinstance(q, dict):
            raise ValueError('Invalid quiz question')
        week = q.get('week')
        if type(week) is not int or not 1 <= week <= 18:
            raise ValueError('Invalid quiz week')
        qid = q.get('id')
        if not isinstance(qid, str) or not re.fullmatch(r'curriculum-w\d{2}-q\d{2}', qid) or qid in seen:
            raise ValueError('Invalid or duplicate quiz ID')
        seen.add(qid)
        for field in ('question', 'explanation', 'category'):
            if not isinstance(q.get(field), str) or not q[field].strip():
                raise ValueError(f'Invalid quiz {field}')
        options = q.get('options')
        if not isinstance(options, list) or not 2 <= len(options) <= 6 or any(not isinstance(o, str) or not o.strip() for o in options) or len(set(options)) != len(options):
            raise ValueError('Invalid quiz options')
        answer = q.get('answer')
        if type(answer) is not int or not 0 <= answer < len(options):
            raise ValueError('Invalid quiz answer')
        if q.get('source') != f'curriculum/week-{week:02d}/lecture.md':
            raise ValueError('Invalid curriculum provenance')
        counts[week] += 1
    if set(counts) != set(range(1, 19)) or min(counts.values()) < MIN_QUESTIONS_PER_WEEK:
        raise ValueError('Quiz seed must cover all 18 weeks with at least 3 questions each')


def load_seed() -> dict:
    bundle = json.loads(SEED_PATH.read_text(encoding='utf-8-sig'))
    validate_seed(bundle)
    return bundle


def seed_quiz_questions(conn: sqlite3.Connection, *, bundle: dict | None = None) -> dict:
    """Insert missing IDs only. Caller owns transaction and connection lifetime.

    IDs remain stable across revisions so teacher edits are never overwritten.
    Existing conflicting IDs are preserved, even when their text differs.
    """
    bundle = load_seed() if bundle is None else bundle
    validate_seed(bundle)  # Validate the entire bundle before the first write.
    inserted = 0
    for q in bundle['questions']:
        cursor = conn.execute(
            'INSERT INTO quiz_questions (id, week, question, options, answer, explanation, category) '
            'VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING',
            (q['id'], q['week'], q['question'], json.dumps(q['options'], ensure_ascii=False),
             q['answer'], q['explanation'], q['category']),
        )
        inserted += cursor.rowcount
    return {'version': bundle['version'], 'inserted': inserted, 'total': len(bundle['questions'])}
