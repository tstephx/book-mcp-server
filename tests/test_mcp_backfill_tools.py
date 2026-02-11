"""Tests for backfill/validate MCP tools."""

import pytest
import sqlite3
import tempfile
import uuid
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)

    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            source_file TEXT,
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


def _insert_library_book(db_path, book_id, title, source_file, chapters=3):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, source_file, word_count) VALUES (?, ?, ?, ?, ?)",
        (book_id, title, "Author", source_file, 10000),
    )
    for i in range(chapters):
        conn.execute(
            "INSERT INTO chapters (id, book_id, title, file_path, word_count, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), book_id, f"Chapter {i+1}", f"ch{i}.md", 3000, b"fake-emb"),
        )
    conn.commit()
    conn.close()


def test_backfill_library_tool_dry_run(db_path):
    from agentic_pipeline.mcp_server import backfill_library

    _insert_library_book(db_path, "book-1", "Book One", "/b.epub")

    result = backfill_library(db_path=str(db_path), dry_run=True)

    assert result["would_backfill"] == 1
    assert result["backfilled"] == 0


def test_backfill_library_tool_execute(db_path):
    from agentic_pipeline.mcp_server import backfill_library
    from agentic_pipeline.db.pipelines import PipelineRepository

    _insert_library_book(db_path, "book-1", "Book One", "/b.epub")

    result = backfill_library(db_path=str(db_path), dry_run=False)

    assert result["backfilled"] == 1
    repo = PipelineRepository(db_path)
    assert repo.get("book-1") is not None


def test_validate_library_tool(db_path):
    from agentic_pipeline.mcp_server import validate_library

    # Book with no chapters = quality issue
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, source_file, word_count) VALUES (?, ?, ?, ?, ?)",
        ("empty-book", "Empty", "Author", "/b.epub", 0),
    )
    conn.commit()
    conn.close()

    result = validate_library(db_path=str(db_path))

    assert result["issue_count"] > 0
    assert any(i["issue"] == "no_chapters" for i in result["issues"])
