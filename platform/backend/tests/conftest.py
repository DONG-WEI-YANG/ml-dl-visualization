"""Shared test fixtures — use temp database for all tests."""
import os
import tempfile
from pathlib import Path

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
