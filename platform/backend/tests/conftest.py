"""Shared test fixtures — use temp database for all tests."""
import os
import tempfile
from pathlib import Path

import pytest

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


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """TestClient requests all share one client IP; the growing test suite
    can exceed the app's rate-limit window across the whole pytest session.
    Reset the shared counter before each test so tests don't pollute each other."""
    from app.main import _rate_limits
    _rate_limits.clear()
    yield
