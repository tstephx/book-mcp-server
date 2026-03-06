"""Tests for the update-title CLI command."""

import sqlite3
import pytest
from click.testing import CliRunner
from agentic_pipeline.cli import main


@pytest.fixture
def db_with_book(tmp_path):
    """Create a test DB with a book and chapters."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER)"
    )
    conn.execute(
        "INSERT INTO books VALUES ('book-1', 'Test Book', 'Author', 1000)"
    )
    conn.execute(
        "INSERT INTO chapters VALUES ('ch-1', 'book-1', 1, 'Old Title', 500)"
    )
    conn.execute(
        "INSERT INTO chapters VALUES ('ch-2', 'book-1', 2, 'Chapter Two', 500)"
    )
    conn.commit()
    conn.close()
    return db_path


def test_update_title_by_chapter_number(db_with_book, monkeypatch):
    """update-title should update the chapter title."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "book-1", "1", "New Title"])
    assert result.exit_code == 0
    assert "New Title" in result.output

    # Verify in DB
    conn = sqlite3.connect(db_with_book)
    row = conn.execute(
        "SELECT title FROM chapters WHERE book_id = 'book-1' AND chapter_number = 1"
    ).fetchone()
    conn.close()
    assert row[0] == "New Title"


def test_update_title_nonexistent_chapter(db_with_book, monkeypatch):
    """update-title should error for nonexistent chapter."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "book-1", "99", "New Title"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "no chapter" in result.output.lower()


def test_update_title_nonexistent_book(db_with_book, monkeypatch):
    """update-title should error for nonexistent book."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "nonexistent", "1", "New Title"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "no book" in result.output.lower()
