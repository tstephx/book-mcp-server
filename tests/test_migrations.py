"""Tests for database migrations."""

import pytest
import sqlite3
import tempfile
from pathlib import Path


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


def test_run_migrations_creates_pipeline_tables(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check processing_pipelines table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='processing_pipelines'"
    )
    assert cursor.fetchone() is not None

    # Check pipeline_state_history table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_state_history'"
    )
    assert cursor.fetchone() is not None

    conn.close()


def test_run_migrations_creates_autonomy_tables(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check autonomy_config table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_config'"
    )
    assert cursor.fetchone() is not None

    # Check it has default row
    cursor.execute("SELECT current_mode FROM autonomy_config WHERE id = 1")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "supervised"

    conn.close()


def test_run_migrations_is_idempotent(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    # Run twice - should not raise
    run_migrations(temp_db)
    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM autonomy_config")
    assert cursor.fetchone()[0] == 1  # Still only one row
    conn.close()
