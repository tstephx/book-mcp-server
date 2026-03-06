"""Tests for the library-issues CLI command."""

import sqlite3
import pytest
from click.testing import CliRunner
from agentic_pipeline.cli import main


@pytest.fixture
def db_with_issues(tmp_path):
    """Create a test DB with various data quality issues."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER)"
    )
    # Empty title
    conn.execute("INSERT INTO books VALUES ('b1', '', 'Author A', 1000)")
    # ISBN-as-title
    conn.execute("INSERT INTO books VALUES ('b2', '1394159641', 'Author B', 2000)")
    # Good book
    conn.execute("INSERT INTO books VALUES ('b3', 'Clean Code', 'Martin', 3000)")
    # Duplicate title
    conn.execute("INSERT INTO books VALUES ('b4', 'Clean Code', 'Martin', 3000)")
    conn.commit()
    conn.close()
    return db_path


def test_library_issues_finds_problems(db_with_issues, monkeypatch):
    """library-issues should report empty titles, ISBN titles, and duplicates."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_issues))
    runner = CliRunner()
    result = runner.invoke(main, ["library-issues"])
    assert result.exit_code == 0
    # Should report empty titles
    assert "empty" in result.output.lower() or "missing" in result.output.lower()
    # Should report duplicates
    assert "duplicate" in result.output.lower()


def test_library_issues_clean_db(tmp_path, monkeypatch):
    """library-issues should report no issues on a clean DB."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute("INSERT INTO books VALUES ('b1', 'Good Book', 'Author', 1000)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
    runner = CliRunner()
    result = runner.invoke(main, ["library-issues"])
    assert result.exit_code == 0
    assert "no issues" in result.output.lower() or "clean" in result.output.lower()
