"""Tests for database connection configuration."""

import tempfile
import sqlite3
from pathlib import Path
import pytest


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_foreign_keys_enabled(db_path):
    """Every connection must enforce foreign key constraints."""
    from agentic_pipeline.db.connection import get_pipeline_db

    with get_pipeline_db(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        assert result[0] == 1, (
            "PRAGMA foreign_keys is OFF — FK constraints not enforced. "
            "Add 'PRAGMA foreign_keys = ON' to get_pipeline_db()."
        )
