"""Tests for library status dashboard."""

import pytest
import sqlite3
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    """Create a temp DB with both library and pipeline tables."""
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)

    # Create pipeline tables
    run_migrations(path)

    # Create library tables (books + chapters)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            word_count INTEGER DEFAULT 0,
            processing_status TEXT DEFAULT 'complete',
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_number INTEGER,
            title TEXT,
            content TEXT,
            word_count INTEGER DEFAULT 0,
            embedding BLOB,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    conn.commit()
    conn.close()

    yield path
    path.unlink(missing_ok=True)


def _add_book(db_path, book_id, title, author=None, word_count=1000,
              chapters=None, pipeline_state=None):
    """Helper to add a book with chapters."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, word_count) VALUES (?, ?, ?, ?)",
        (book_id, title, author, word_count),
    )
    if chapters:
        for ch_num, has_embedding in enumerate(chapters, 1):
            ch_id = f"{book_id}-ch{ch_num}"
            embedding = b"fake-embedding" if has_embedding else None
            conn.execute(
                "INSERT INTO chapters (id, book_id, chapter_number, title, embedding) VALUES (?, ?, ?, ?, ?)",
                (ch_id, book_id, ch_num, f"Chapter {ch_num}", embedding),
            )
    if pipeline_state:
        conn.execute(
            "INSERT OR IGNORE INTO processing_pipelines (id, source_path, content_hash, state) VALUES (?, ?, ?, ?)",
            (book_id, f"/books/{book_id}.epub", f"hash-{book_id}", pipeline_state),
        )
    conn.commit()
    conn.close()


def test_empty_library(db_path):
    from agentic_pipeline.library import LibraryStatus

    monitor = LibraryStatus(db_path)
    result = monitor.get_status()

    overview = result["overview"]
    assert overview["total_books"] == 0
    assert overview["total_chapters"] == 0
    assert overview["total_words"] == 0
    assert overview["books_fully_ready"] == 0
    assert overview["books_partially_ready"] == 0
    assert overview["books_not_embedded"] == 0
    assert overview["embedded_chapters"] == 0
    assert overview["embedding_coverage_pct"] == 0.0
    assert result["books"] == []


def test_fully_embedded_book(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "book1", "Test Book", chapters=[True, True, True])

    result = LibraryStatus(db_path).get_status()
    book = result["books"][0]

    assert book["status"] == "ready"
    assert book["chapters"] == 3
    assert book["embedded_chapters"] == 3
    assert book["embedding_pct"] == 100.0


def test_partially_embedded_book(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "book1", "Test Book", chapters=[True, False, True])

    result = LibraryStatus(db_path).get_status()
    book = result["books"][0]

    assert book["status"] == "partial"
    assert book["embedded_chapters"] == 2
    assert book["embedding_pct"] == pytest.approx(66.7, abs=0.1)


def test_no_embeddings_book(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "book1", "Test Book", chapters=[False, False])

    result = LibraryStatus(db_path).get_status()
    book = result["books"][0]

    assert book["status"] == "no_embeddings"
    assert book["embedded_chapters"] == 0


def test_book_with_pipeline_record(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "book1", "Pipeline Book",
              chapters=[True], pipeline_state="complete")

    result = LibraryStatus(db_path).get_status()
    book = result["books"][0]

    assert book["source"] == "pipeline"
    assert book["pipeline_state"] == "complete"


def test_book_without_pipeline_record(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "book1", "Direct Book", chapters=[True])

    result = LibraryStatus(db_path).get_status()
    book = result["books"][0]

    assert book["source"] == "direct"
    assert book["pipeline_state"] is None


def test_overview_counts(db_path):
    from agentic_pipeline.library import LibraryStatus

    _add_book(db_path, "b1", "Ready Book", word_count=5000, chapters=[True, True])
    _add_book(db_path, "b2", "Partial Book", word_count=3000, chapters=[True, False])
    _add_book(db_path, "b3", "No Embed Book", word_count=2000, chapters=[False, False, False])

    result = LibraryStatus(db_path).get_status()
    overview = result["overview"]

    assert overview["total_books"] == 3
    assert overview["total_chapters"] == 7
    assert overview["total_words"] == 10000
    assert overview["books_fully_ready"] == 1
    assert overview["books_partially_ready"] == 1
    assert overview["books_not_embedded"] == 1
    assert overview["embedded_chapters"] == 3
    assert overview["embedding_coverage_pct"] == pytest.approx(42.9, abs=0.1)


def test_pipeline_summary_state_counts(db_path):
    from agentic_pipeline.library import LibraryStatus
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    repo.create("/a.epub", "ha", priority=5)
    repo.create("/b.epub", "hb", priority=5)

    result = LibraryStatus(db_path).get_status()
    pipeline = result["pipeline_summary"]

    assert pipeline["total_pipelines"] == 2
    assert "detected" in pipeline["by_state"]
    assert pipeline["by_state"]["detected"] == 2


def test_missing_tables():
    """When library tables don't exist, returns empty status."""
    from agentic_pipeline.library import LibraryStatus

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)

    try:
        result = LibraryStatus(path).get_status()
        assert result["overview"]["total_books"] == 0
        assert result["books"] == []
    finally:
        path.unlink(missing_ok=True)


def test_cli_command_exists():
    from click.testing import CliRunner
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["library-status", "--help"])
    assert result.exit_code == 0
    assert "library status dashboard" in result.output.lower()
