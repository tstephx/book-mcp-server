"""Tests for chunks table migration."""

import sqlite3
import tempfile
from pathlib import Path

from agentic_pipeline.db.migrations import run_migrations


class TestChunksMigration:
    def test_chunks_table_created(self):
        """run_migrations creates the chunks table."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            )
            assert cursor.fetchone() is not None
            conn.close()

    def test_chunks_table_columns(self):
        """chunks table has all expected columns."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chunks)")
            columns = {row[1] for row in cursor.fetchall()}

            expected = {
                "id", "chapter_id", "book_id", "chunk_index",
                "content", "word_count", "embedding", "embedding_model",
                "content_hash", "created_at",
            }
            assert expected.issubset(columns)
            conn.close()

    def test_chunks_indexes_created(self):
        """Chunk lookup indexes are created."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_chunks%'"
            )
            indexes = {row[0] for row in cursor.fetchall()}
            assert "idx_chunks_chapter" in indexes
            assert "idx_chunks_book" in indexes
            conn.close()
