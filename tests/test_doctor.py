"""Tests for the integrity doctor."""

import sqlite3

import pytest

from agentic_pipeline.db.migrations import run_migrations


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "doctor.db"
    run_migrations(path)
    # NOTE (adjustment): run_migrations() only creates agentic_pipeline's own
    # tables (processing_pipelines, chunks, etc.) — `books`/`chapters` are
    # created by the external book-ingestion package against the real,
    # shared library DB and don't exist in a fresh migrated test DB. Several
    # existing test files (test_backfill.py, test_library_status.py,
    # test_mcp_backfill_tools.py, test_reprocess_cli.py) already work around
    # this the same way: create the minimal books/chapters schema ad hoc.
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            book_type TEXT,
            source_file TEXT,
            word_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_number INTEGER,
            title TEXT,
            file_path TEXT,
            word_count INTEGER DEFAULT 0,
            content_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )"""
    )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Seed helpers — raw sqlite3 is fine in tests; ids are TEXT PKs in all tables.
# ---------------------------------------------------------------------------


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_book(conn, book_id="b1", title="Live Book", book_type=None):
    conn.execute(
        "INSERT INTO books (id, title, book_type) VALUES (?, ?, ?)",
        (book_id, title, book_type),
    )


def _seed_chapter(conn, chapter_id="c1", book_id="b1", file_path="/nope.md", content_hash="hash", number=1):
    conn.execute(
        """INSERT INTO chapters (id, book_id, chapter_number, title, file_path,
                                 word_count, content_hash)
           VALUES (?, ?, ?, ?, ?, 100, ?)""",
        (chapter_id, book_id, number, f"Chapter {number}", file_path, content_hash),
    )


def _seed_chunk(
    conn, chunk_id="k1", chapter_id="c1", book_id="b1", content="chunk text body", embedded=True, chunk_index=0
):
    # chunk_index defaults to 0 for every existing call site; only pass a
    # non-default value when seeding >1 chunk under the same chapter_id —
    # chunks has a UNIQUE INDEX on (chapter_id, chunk_index).
    conn.execute(
        """INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content,
                               word_count, embedding, embedding_model, content_hash)
           VALUES (?, ?, ?, ?, ?, 3, ?, 'test-model', 'kh')""",
        (chunk_id, chapter_id, book_id, chunk_index, content, b"\x00\x01" if embedded else None),
    )


class TestCheckOrphanedChunks:
    def test_flags_chunk_with_dead_chapter_id(self, db_path):
        """Seeded violation — the check MUST be able to fail."""
        from agentic_pipeline.health.doctor import (
            CATEGORY_ORPHANED_CHUNKS,
            check_orphaned_chunks,
        )

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="ok", chapter_id="c1")  # healthy
        _seed_chunk(conn, chunk_id="orphan", chapter_id="GONE")  # dead chapter
        conn.commit()
        conn.close()

        finding = check_orphaned_chunks(db_path)

        assert finding.category == CATEGORY_ORPHANED_CHUNKS
        assert finding.count == 1
        assert finding.fixable_count == 1
        assert finding.details[0]["chunk_id"] == "orphan"

    def test_flags_chunk_with_dead_book_id(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="orphan2", book_id="GONE-BOOK")
        conn.commit()
        conn.close()

        assert check_orphaned_chunks(db_path).count == 1

    def test_counts_unembedded_orphans_identically(self, db_path):
        """Spec: doctor treats embedded and never-embedded orphans identically."""
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o1", chapter_id="GONE", book_id="GONE", embedded=True, chunk_index=0)
        _seed_chunk(conn, chunk_id="o2", chapter_id="GONE", book_id="GONE", embedded=False, chunk_index=1)
        conn.commit()
        conn.close()

        assert check_orphaned_chunks(db_path).count == 2

    def test_clean_db_passes(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn)
        conn.commit()
        conn.close()

        finding = check_orphaned_chunks(db_path)
        assert finding.count == 0
        assert finding.details == []

    def test_finding_field_types(self, db_path):
        """Contract rule: assert types, not just presence."""
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        finding = check_orphaned_chunks(db_path)
        assert isinstance(finding.category, str)
        assert isinstance(finding.count, int)
        assert isinstance(finding.fixable_count, int)
        assert isinstance(finding.details, list)
