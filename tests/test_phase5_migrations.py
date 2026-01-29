# tests/test_phase5_migrations.py
"""Tests for Phase 5 database migrations."""

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


def test_autonomy_config_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_config'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_thresholds_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_thresholds'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_feedback_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_feedback'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_spot_checks_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='spot_checks'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_config_has_default_row(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT current_mode FROM autonomy_config WHERE id = 1")
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == "supervised"
