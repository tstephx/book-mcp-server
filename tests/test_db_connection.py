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


def test_mcp_tool_connection_foreign_keys_enabled(db_path, monkeypatch):
    """src.database.get_db_connection() — used by every MCP tool in
    src/tools/ — must also enforce foreign keys, not just the pipeline-side
    get_pipeline_db(). Both connect to the same library.db by default; if
    only one enables PRAGMA foreign_keys, ON DELETE CASCADE silently doesn't
    fire on the other (book-mcp-server#9)."""
    from src.config import Config

    monkeypatch.setattr(Config, "DB_PATH", db_path)

    from src.database import get_db_connection

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        assert result[0] == 1, (
            "PRAGMA foreign_keys is OFF — FK constraints not enforced. "
            "Add 'PRAGMA foreign_keys = ON' to get_db_connection()."
        )


@pytest.fixture
def books_chapters_db_path():
    """Minimal standalone books/chapters schema with the same ON DELETE
    CASCADE relationship the real library.db declares — self-contained so
    this test doesn't depend on the multi-repo migration chain (books'/
    chapters' base columns actually come from book-ingestion-python, not
    this repo) just to exercise a PRAGMA foreign_keys behavior."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT);
        CREATE TABLE chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_number INTEGER,
            title TEXT,
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()
    yield path
    path.unlink(missing_ok=True)


def test_mcp_tool_connection_cascade_deletes(books_chapters_db_path, monkeypatch):
    """End-to-end: deleting a book through the MCP-tool connection path must
    cascade to its chapters, not orphan them (book-mcp-server#9)."""
    from src.config import Config

    monkeypatch.setattr(Config, "DB_PATH", books_chapters_db_path)

    from src.database import get_db_connection

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO books (id, title, author) VALUES (?, ?, ?)",
            ("book-1", "Test Book", "Test Author"),
        )
        conn.execute(
            "INSERT INTO chapters (id, book_id, chapter_number, title) VALUES (?, ?, ?, ?)",
            ("chapter-1", "book-1", 1, "Chapter One"),
        )
        conn.commit()

        conn.execute("DELETE FROM books WHERE id = ?", ("book-1",))
        conn.commit()

        remaining = conn.execute("SELECT COUNT(*) FROM chapters WHERE book_id = ?", ("book-1",)).fetchone()[0]
        assert remaining == 0, (
            "Deleting a book left orphaned chapter rows — ON DELETE CASCADE did not fire on this connection."
        )
