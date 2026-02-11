"""Tests for pipeline backfill."""

import pytest
import sqlite3
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)

    # Create library tables (not in pipeline migrations)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            file_path TEXT,
            word_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT,
            title TEXT,
            file_path TEXT,
            word_count INTEGER DEFAULT 0,
            embedding BLOB,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    conn.commit()
    conn.close()
    yield path
    path.unlink(missing_ok=True)


def test_create_backfill_inserts_at_complete(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    repo.create_backfill(
        book_id="existing-book-123",
        source_path="/books/test.epub",
        content_hash="abc123def456",
    )

    record = repo.get("existing-book-123")
    assert record is not None
    assert record["state"] == PipelineState.COMPLETE.value
    assert record["source_path"] == "/books/test.epub"
    assert record["content_hash"] == "abc123def456"
    assert record["approved_by"] == "backfill:automated"


def test_create_backfill_skips_existing(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)

    # Create a normal pipeline record first
    pid = repo.create("/books/test.epub", "hash123")

    # Backfill with same hash should return False (skip)
    result = repo.create_backfill(
        book_id="different-id",
        source_path="/books/test.epub",
        content_hash="hash123",
    )
    assert result is False


def test_create_backfill_uses_book_id_as_pipeline_id(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    repo.create_backfill(
        book_id="my-uuid-here",
        source_path="/books/test.epub",
        content_hash="unique-hash",
    )

    # Pipeline ID should be the book_id, not a generated UUID
    record = repo.get("my-uuid-here")
    assert record is not None
    assert record["id"] == "my-uuid-here"
