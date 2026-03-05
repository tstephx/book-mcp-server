"""Tests for schema_migrations version tracking."""

import sqlite3
import tempfile
from pathlib import Path

from agentic_pipeline.db.migrations import run_migrations


def _open(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class TestSchemaMigrationsTable:
    def test_schema_migrations_table_exists(self):
        """run_migrations creates schema_migrations tracking table."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            run_migrations(Path(f.name))
            conn = _open(f.name)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            )
            assert cursor.fetchone() is not None, "schema_migrations table not created"
            conn.close()

    def test_applied_migrations_recorded(self):
        """Known ALTER TABLE migrations are recorded in schema_migrations."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            run_migrations(Path(f.name))
            conn = _open(f.name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM schema_migrations")
            names = {row["name"] for row in cursor.fetchall()}
            conn.close()

            assert "add_processing_result_column" in names, (
                "Expected 'add_processing_result_column' to be recorded in schema_migrations"
            )

    def test_run_migrations_idempotent(self):
        """Running migrations twice on same DB is safe — no errors, no duplicate records."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            run_migrations(Path(f.name))
            run_migrations(Path(f.name))  # must not raise

            conn = _open(f.name)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM schema_migrations WHERE name = 'add_processing_result_column'"
            )
            count = cursor.fetchone()["cnt"]
            conn.close()

            assert count == 1, f"Expected 1 record, got {count} — migration recorded multiple times"
