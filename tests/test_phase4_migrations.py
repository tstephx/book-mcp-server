# tests/test_phase4_migrations.py
"""Tests for Phase 4 database migrations."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_priority_column_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(processing_pipelines)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "priority" in columns


def test_audit_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='approval_audit'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_health_metrics_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='health_metrics'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_state_duration_stats_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='state_duration_stats'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None
