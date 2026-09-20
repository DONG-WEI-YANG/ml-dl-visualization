import copy
import sqlite3
from collections import Counter
from pathlib import Path

import pytest


def test_bundle_covers_curriculum_with_valid_answers():
    from app.quiz.seed import load_seed, MIN_QUESTIONS_PER_WEEK
    bundle = load_seed()
    assert bundle['version']
    counts = Counter(q['week'] for q in bundle['questions'])
    assert set(counts) == set(range(1, 19))
    assert min(counts.values()) >= MIN_QUESTIONS_PER_WEEK
    root = Path(__file__).resolve().parents[3]
    for q in bundle['questions']:
        assert (root / q['source']).is_file()
        assert q['options'][q['answer']]


def test_seed_is_additive_idempotent_and_transaction_owned_by_caller():
    from app.quiz.seed import seed_quiz_questions, load_seed
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE quiz_questions (id TEXT PRIMARY KEY, week INTEGER, question TEXT, options TEXT, answer INTEGER, explanation TEXT, category TEXT)')
    assert seed_quiz_questions(conn)['inserted'] == 54
    first = load_seed()['questions'][0]['id']
    conn.execute('UPDATE quiz_questions SET question = ? WHERE id = ?', ('Teacher edit', first))
    conn.execute("INSERT INTO quiz_questions VALUES ('custom', 1, 'Custom', '[\"A\",\"B\"]', 0, '', 'concept')")
    assert seed_quiz_questions(conn)['inserted'] == 0
    assert conn.execute('SELECT question FROM quiz_questions WHERE id = ?', (first,)).fetchone()[0] == 'Teacher edit'
    assert conn.execute('SELECT count(*) FROM quiz_questions').fetchone()[0] == 55
    conn.rollback()
    assert conn.execute('SELECT count(*) FROM quiz_questions').fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize('damage', ['answer', 'duplicate', 'missing_week', 'options', 'week', 'source'])
def test_invalid_seed_rejected_before_any_insert(damage):
    from app.quiz.seed import load_seed, seed_quiz_questions
    bundle = copy.deepcopy(load_seed())
    q = bundle['questions'][0]
    if damage == 'answer': q['answer'] = True
    if damage == 'duplicate': bundle['questions'].append(copy.deepcopy(q))
    if damage == 'missing_week': bundle['questions'] = [q for q in bundle['questions'] if q['week'] != 18]
    if damage == 'options': q['options'] = ['A', 'A']
    if damage == 'week': q['week'] = 19
    if damage == 'source': q['source'] = '../outside.md'
    conn = sqlite3.connect(':memory:')
    with pytest.raises(ValueError):
        seed_quiz_questions(conn, bundle=bundle)
    assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
    conn.close()
