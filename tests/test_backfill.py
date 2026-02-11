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


def _insert_library_book(db_path, book_id, title, file_path, chapters=3):
    """Insert a fake library book with chapters."""
    import uuid
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, file_path, word_count) VALUES (?, ?, ?, ?, ?)",
        (book_id, title, "Test Author", file_path, 10000),
    )
    for i in range(chapters):
        conn.execute(
            "INSERT INTO chapters (id, book_id, title, file_path, word_count, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), book_id, f"Chapter {i+1}", f"ch{i}.md", 3000, b"fake-emb"),
        )
    conn.commit()
    conn.close()


def test_backfill_manager_finds_untracked_books(db_path):
    from agentic_pipeline.backfill import BackfillManager

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")
    _insert_library_book(db_path, "book-2", "Book Two", "/books/two.epub")

    manager = BackfillManager(db_path)
    untracked = manager.find_untracked()

    assert len(untracked) == 2
    assert untracked[0]["id"] in ("book-1", "book-2")


def test_backfill_manager_skips_tracked_books(db_path):
    from agentic_pipeline.backfill import BackfillManager
    from agentic_pipeline.db.pipelines import PipelineRepository

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")

    repo = PipelineRepository(db_path)
    repo.create_backfill("book-1", "/books/one.epub", "somehash")

    manager = BackfillManager(db_path)
    untracked = manager.find_untracked()

    assert len(untracked) == 0


def test_backfill_manager_run_creates_records(db_path):
    from agentic_pipeline.backfill import BackfillManager
    from agentic_pipeline.db.pipelines import PipelineRepository

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")
    _insert_library_book(db_path, "book-2", "Book Two", "/books/two.epub")

    manager = BackfillManager(db_path)
    result = manager.run()

    assert result["backfilled"] == 2
    assert result["skipped"] == 0

    repo = PipelineRepository(db_path)
    assert repo.get("book-1") is not None
    assert repo.get("book-2") is not None


def test_backfill_manager_dry_run(db_path):
    from agentic_pipeline.backfill import BackfillManager
    from agentic_pipeline.db.pipelines import PipelineRepository

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")

    manager = BackfillManager(db_path)
    result = manager.run(dry_run=True)

    assert result["backfilled"] == 0
    assert result["would_backfill"] == 1

    repo = PipelineRepository(db_path)
    assert repo.get("book-1") is None


def test_backfill_manager_includes_quality_stats(db_path):
    from agentic_pipeline.backfill import BackfillManager

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub", chapters=5)

    manager = BackfillManager(db_path)
    result = manager.run()

    assert result["backfilled"] == 1
    book = result["books"][0]
    assert book["chapter_count"] == 5
    assert book["embedded_count"] == 5
    assert book["quality"] == "good"


def test_reingest_archives_old_record(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    repo.create_backfill("book-1", "/books/one.epub", "backfill:hash1")

    # Reingest should archive the old record and return a new pipeline_id
    new_pid = repo.prepare_reingest("book-1")

    # Old record should be archived
    old = repo.get("book-1")
    assert old["state"] == PipelineState.ARCHIVED.value

    # New record should exist in DETECTED state
    new = repo.get(new_pid)
    assert new is not None
    assert new["state"] == PipelineState.DETECTED.value
    assert new["source_path"] == "/books/one.epub"


def test_reingest_not_found(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)

    with pytest.raises(ValueError, match="not found"):
        repo.prepare_reingest("nonexistent-id")


def test_backfill_cli_dry_run(db_path, monkeypatch):
    from click.testing import CliRunner
    from agentic_pipeline.cli import main

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")

    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))

    runner = CliRunner()
    result = runner.invoke(main, ["backfill", "--dry-run"])

    assert result.exit_code == 0
    assert "would backfill" in result.output.lower() or "Would backfill" in result.output


def test_backfill_cli_execute(db_path, monkeypatch):
    from click.testing import CliRunner
    from agentic_pipeline.cli import main
    from agentic_pipeline.db.pipelines import PipelineRepository

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")

    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))

    runner = CliRunner()
    result = runner.invoke(main, ["backfill", "--execute"])

    assert result.exit_code == 0
    assert "1" in result.output  # backfilled count

    repo = PipelineRepository(db_path)
    assert repo.get("book-1") is not None


def test_backfill_creates_audit_entries(db_path):
    from agentic_pipeline.backfill import BackfillManager

    _insert_library_book(db_path, "book-1", "Book One", "/books/one.epub")

    manager = BackfillManager(db_path)
    manager.run()

    # Check audit trail
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM approval_audit WHERE book_id = ? AND action = ?",
        ("book-1", "backfill"),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["actor"] == "backfill:automated"


def test_validator_finds_missing_embeddings(db_path):
    from agentic_pipeline.backfill import LibraryValidator
    import uuid

    # Book with 3 chapters, only 1 has embedding
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, file_path, word_count) VALUES (?, ?, ?, ?, ?)",
        ("book-partial", "Partial Book", "Author", "/book.epub", 10000),
    )
    for i in range(3):
        emb = b"fake" if i == 0 else None
        conn.execute(
            "INSERT INTO chapters (id, book_id, title, file_path, word_count, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "book-partial", f"Ch {i+1}", f"ch{i}.md", 3000, emb),
        )
    conn.commit()
    conn.close()

    validator = LibraryValidator(db_path)
    issues = validator.validate()

    book_issues = [i for i in issues if i["book_id"] == "book-partial"]
    assert len(book_issues) > 0
    assert any(i["issue"] == "missing_embeddings" for i in book_issues)


def test_validator_flags_no_chapters(db_path):
    from agentic_pipeline.backfill import LibraryValidator

    # Book with 0 chapters
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO books (id, title, author, file_path, word_count) VALUES (?, ?, ?, ?, ?)",
        ("book-empty", "Empty Book", "Author", "/book.epub", 0),
    )
    conn.commit()
    conn.close()

    validator = LibraryValidator(db_path)
    issues = validator.validate()

    book_issues = [i for i in issues if i["book_id"] == "book-empty"]
    assert any(i["issue"] == "no_chapters" for i in book_issues)


def test_validator_clean_book_has_no_issues(db_path):
    from agentic_pipeline.backfill import LibraryValidator

    _insert_library_book(db_path, "book-good", "Good Book", "/book.epub", chapters=5)

    validator = LibraryValidator(db_path)
    issues = validator.validate()

    book_issues = [i for i in issues if i["book_id"] == "book-good"]
    assert len(book_issues) == 0
