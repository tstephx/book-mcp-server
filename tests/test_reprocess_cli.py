"""Tests for reprocess CLI command."""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner

from agentic_pipeline.cli import main


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def db_with_books(db_path):
    """Create a DB with books and chapters for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY, title TEXT, author TEXT, source_file TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER,
            title TEXT, word_count INTEGER, content_hash TEXT,
            file_path TEXT, embedding BLOB, embedding_model TEXT
        )
    """)
    # Good book: 10 chapters, ~5k words each
    conn.execute("INSERT INTO books VALUES ('good-1', 'Good Book', 'Author', '/tmp/good.epub')")
    for i in range(1, 11):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'good-1', ?, ?, 5000, ?, '', NULL, NULL)",
            (f"good-1-ch{i}", i, f"Chapter {i}", f"hash-good-{i}"),
        )

    # Bad book: 2 chapters, huge (fails min chapters check)
    conn.execute("INSERT INTO books VALUES ('bad-1', 'Bad Book', 'Author', '/tmp/bad.epub')")
    for i in range(1, 3):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'bad-1', ?, ?, 25000, ?, '', NULL, NULL)",
            (f"bad-1-ch{i}", i, f"Chapter {i}", f"hash-bad-{i}"),
        )

    conn.commit()
    conn.close()
    return db_path


def test_reprocess_requires_flagged(db_with_books):
    """Without --flagged, command shows error."""
    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess"])
    assert result.exit_code == 0
    assert "flagged" in result.output.lower()


def test_reprocess_dry_run(db_with_books):
    """Dry run shows flagged books without modifying anything."""
    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "Bad Book" in result.output


def test_reprocess_dry_run_json(db_with_books):
    """Dry run with --json outputs valid JSON."""
    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged", "--json"])
    assert result.exit_code == 0
    output = result.output.strip()
    data = json.loads(output)
    assert data["flagged"] >= 1
    assert data["mode"] == "dry_run"


def test_reprocess_no_flagged_books(db_path):
    """When all books pass, shows success message."""
    # Create only good books
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY, title TEXT, author TEXT, source_file TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER,
            title TEXT, word_count INTEGER, content_hash TEXT,
            file_path TEXT, embedding BLOB, embedding_model TEXT
        )
    """)
    conn.execute("INSERT INTO books VALUES ('good-1', 'Good Book', 'Author', '/tmp/good.epub')")
    for i in range(1, 11):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'good-1', ?, ?, 5000, ?, '', NULL, NULL)",
            (f"good-1-ch{i}", i, f"Chapter {i}", f"hash-good-{i}"),
        )
    conn.commit()
    conn.close()

    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_path)):
        result = runner.invoke(main, ["reprocess", "--flagged"])
    assert result.exit_code == 0
    assert "pass" in result.output.lower() or "0" in result.output


def test_reprocess_execute_missing_source(db_with_books):
    """Execute mode skips books with missing source files."""
    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged", "--execute"])
    assert result.exit_code == 0
    assert "skipping" in result.output.lower() or "skip" in result.output.lower()


def test_reprocess_execute_with_source(db_with_books, tmp_path):
    """Execute mode re-queues books when source file exists."""
    # Create a fake source file
    source_file = tmp_path / "bad.epub"
    source_file.write_bytes(b"fake epub content")

    # Update the bad book to point to the real source file
    conn = sqlite3.connect(str(db_with_books))
    conn.execute("UPDATE books SET source_file = ? WHERE id = 'bad-1'", (str(source_file),))
    conn.commit()
    conn.close()

    runner = CliRunner()
    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged", "--execute"])
    assert result.exit_code == 0
    assert "re-queued" in result.output.lower()

    # Verify old book record was deleted and pipeline entry created
    conn = sqlite3.connect(str(db_with_books))
    conn.row_factory = sqlite3.Row
    old_book = conn.execute("SELECT * FROM books WHERE id = 'bad-1'").fetchone()
    assert old_book is None

    old_chapters = conn.execute("SELECT * FROM chapters WHERE book_id = 'bad-1'").fetchall()
    assert len(old_chapters) == 0

    pipelines = conn.execute("SELECT * FROM processing_pipelines WHERE source_path = ?", (str(source_file),)).fetchall()
    assert len(pipelines) == 1
    assert pipelines[0]["state"] == "detected"
    conn.close()
