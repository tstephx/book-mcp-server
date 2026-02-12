"""Tests for ensure_library_schema() — library-side auto-migration."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def empty_db():
    """Create a temp DB with only books/chapters (simulating book-ingestion base)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            word_count INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            title TEXT,
            file_path TEXT NOT NULL,
            word_count INTEGER,
            embedding BLOB,
            embedding_model TEXT,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    conn.commit()
    conn.close()

    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def full_db(empty_db):
    """DB that already has all library tables (simulating existing install)."""
    conn = sqlite3.connect(str(empty_db))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE VIRTUAL TABLE chapters_fts USING fts5(
            chapter_id, title, content,
            tokenize='porter unicode61'
        )
    """)
    cursor.execute("""
        CREATE TABLE chapter_summaries (
            chapter_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            summary_type TEXT DEFAULT 'extractive',
            word_count INTEGER,
            generated_at TEXT,
            embedding BLOB,
            embedding_model TEXT,
            FOREIGN KEY (chapter_id) REFERENCES chapters(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE reading_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            status TEXT DEFAULT 'unread',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (book_id) REFERENCES books(id),
            UNIQUE(book_id, chapter_number)
        )
    """)
    cursor.execute("""
        CREATE TABLE bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            position INTEGER DEFAULT 0,
            title TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    # Also add tracking columns to chapters
    for col in ["content_hash TEXT", "file_mtime REAL", "embedding_updated_at TEXT"]:
        try:
            cursor.execute(f"ALTER TABLE chapters ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

    yield empty_db


def _table_exists(cursor, name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cursor.fetchone() is not None


def _column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    return column in [row[1] for row in cursor.fetchall()]


class TestEnsureLibrarySchemaFreshDB:
    """On a fresh DB with only books/chapters, everything should be created."""

    def test_creates_chapters_fts(self, empty_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _table_exists(cursor, "chapters_fts")
        conn.close()

    def test_creates_chapter_summaries(self, empty_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _table_exists(cursor, "chapter_summaries")
        assert _column_exists(cursor, "chapter_summaries", "embedding")
        assert _column_exists(cursor, "chapter_summaries", "embedding_model")
        conn.close()

    def test_creates_reading_progress(self, empty_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _table_exists(cursor, "reading_progress")
        conn.close()

    def test_creates_bookmarks(self, empty_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _table_exists(cursor, "bookmarks")
        conn.close()

    def test_adds_tracking_columns_to_chapters(self, empty_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _column_exists(cursor, "chapters", "content_hash")
        assert _column_exists(cursor, "chapters", "file_mtime")
        assert _column_exists(cursor, "chapters", "embedding_updated_at")
        conn.close()


class TestEnsureLibrarySchemaIdempotent:
    """On a DB that already has everything, function is a no-op."""

    def test_idempotent_no_error(self, full_db):
        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = full_db
            from src.database import ensure_library_schema
            ensure_library_schema()
            # Call twice — should not raise
            ensure_library_schema()

    def test_preserves_existing_data(self, full_db):
        # Insert test data before running
        conn = sqlite3.connect(str(full_db))
        conn.execute(
            "INSERT INTO chapter_summaries (chapter_id, summary) VALUES ('ch1', 'test summary')"
        )
        conn.commit()
        conn.close()

        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = full_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(full_db))
        row = conn.execute(
            "SELECT summary FROM chapter_summaries WHERE chapter_id = 'ch1'"
        ).fetchone()
        assert row[0] == "test summary"
        conn.close()


class TestEnsureLibrarySchemaPartialState:
    """DB with some tables but missing columns (like prod before this fix)."""

    def test_adds_missing_embedding_columns(self, empty_db):
        # Create chapter_summaries WITHOUT embedding columns (prod state)
        conn = sqlite3.connect(str(empty_db))
        conn.execute("""
            CREATE TABLE chapter_summaries (
                chapter_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summary_type TEXT DEFAULT 'extractive',
                word_count INTEGER,
                generated_at TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            )
        """)
        conn.commit()
        conn.close()

        with patch("src.database.Config") as mock_config:
            mock_config.DB_PATH = empty_db
            from src.database import ensure_library_schema
            ensure_library_schema()

        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        assert _column_exists(cursor, "chapter_summaries", "embedding")
        assert _column_exists(cursor, "chapter_summaries", "embedding_model")
        conn.close()
