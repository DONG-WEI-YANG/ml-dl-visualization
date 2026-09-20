"""Shared test fixtures — use temp database for all tests."""
import os
import tempfile
from pathlib import Path

import pytest

# Explicit test configuration; never inherit a workstation's production secrets.
os.environ['APP_ENV'] = 'test'
os.environ['JWT_SECRET'] = 'isolated-test-signing-secret-0123456789'
os.environ['DEFAULT_ADMIN_PASSWORD'] = 'admin123'

# Create temp DB before any app imports
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_test_db.close()

# Override DB_PATH before importing app modules
import app.db as db_module
db_module.DB_PATH = Path(_test_db.name)

# Initialize database tables
from app.db import init_db
from app.rag.store import init_rag_tables

init_db()
init_rag_tables()

# Existing API tests use an already-onboarded administrator. Forced-change
# behavior is exercised separately with accounts that retain the flag.
with db_module.db_connection() as conn:
    conn.execute("UPDATE users SET must_change_password = 0 WHERE username = 'admin'")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """TestClient requests all share one client IP; the growing test suite
    can exceed the app's rate-limit window across the whole pytest session.
    Reset the shared counter before each test so tests don't pollute each other."""
    from app.main import _rate_limits
    _rate_limits.clear()
    yield
