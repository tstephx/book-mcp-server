"""Tests for summary embeddings migration, generation, and caching"""

import io
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from migrations.add_summary_embeddings import add_summary_embeddings, column_exists


# ─── Migration Tests ──────────────────────────────────────────────────

@pytest.fixture
def db_with_summaries():
    """Create a temp DB with chapter_summaries table (no embedding columns)"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    # Create minimal books/chapters tables
    cursor.execute("""
        CREATE TABLE books (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            word_count INTEGER DEFAULT 0,
            processing_status TEXT DEFAULT 'complete',
            added_date TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT,
            chapter_number INTEGER,
            title TEXT,
            file_path TEXT,
            word_count INTEGER DEFAULT 0,
            embedding BLOB,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE chapter_summaries (
            chapter_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            summary_type TEXT DEFAULT 'extractive',
            word_count INTEGER,
            generated_at TEXT,
            FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
        )
    """)

    # Insert test data
    cursor.execute(
        "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)",
        ('book-1', 'Test Book', 'Author', 10000, 'complete', '2024-01-01')
    )
    cursor.execute(
        "INSERT INTO chapters VALUES (?, ?, ?, ?, ?, ?, ?)",
        ('ch-1', 'book-1', 1, 'Chapter One', '/tmp/ch1.md', 5000, None)
    )
    cursor.execute(
        "INSERT INTO chapter_summaries VALUES (?, ?, ?, ?, ?)",
        ('ch-1', 'This is a test summary.', 'extractive', 5, '2024-01-01T00:00:00')
    )

    conn.commit()
    conn.close()

    yield path
    path.unlink(missing_ok=True)


def test_migration_adds_columns(db_with_summaries):
    """Migration should add embedding and embedding_model columns"""
    result = add_summary_embeddings(db_with_summaries)

    assert result['columns_added'] == 2
    assert result['status'] == 'success'

    # Verify columns exist
    conn = sqlite3.connect(str(db_with_summaries))
    cursor = conn.cursor()

    assert column_exists(cursor, 'chapter_summaries', 'embedding')
    assert column_exists(cursor, 'chapter_summaries', 'embedding_model')

    conn.close()


def test_migration_idempotent(db_with_summaries):
    """Running migration twice should not fail"""
    result1 = add_summary_embeddings(db_with_summaries)
    assert result1['columns_added'] == 2

    result2 = add_summary_embeddings(db_with_summaries)
    assert result2['columns_added'] == 0


def test_migration_no_summaries_table():
    """Migration should fail gracefully without chapter_summaries table"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)

    # Create empty DB
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE books (id TEXT)")
    conn.commit()
    conn.close()

    try:
        result = add_summary_embeddings(path)
        assert 'error' in result
    finally:
        path.unlink(missing_ok=True)


# ─── Summary Embedding Generation Tests ──────────────────────────────

@pytest.fixture
def db_with_embedding_columns(db_with_summaries):
    """DB with summaries table that has embedding columns"""
    add_summary_embeddings(db_with_summaries)
    return db_with_summaries


def _make_mock_generator():
    """Create a mock embedding generator"""
    gen = MagicMock()
    gen.model_name = 'test-model'
    gen.generate.return_value = np.random.randn(384).astype(np.float32)
    return gen


def test_generate_single_embedding(db_with_embedding_columns):
    """generate_summary_embedding should store BLOB in DB"""
    db_path = db_with_embedding_columns
    gen = _make_mock_generator()

    with patch('src.utils.summaries.execute_query') as mock_query, \
         patch('src.utils.summaries.get_db_connection') as mock_conn_ctx:

        # Mock the summary lookup
        mock_query.return_value = [{'summary': 'This is a test summary.'}]

        # Mock DB connection for storing
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        from src.utils.summaries import generate_summary_embedding
        result = generate_summary_embedding('ch-1', gen)

        assert result is True
        gen.generate.assert_called_once_with('This is a test summary.')
        mock_cursor.execute.assert_called_once()

        # Verify the BLOB was passed
        call_args = mock_cursor.execute.call_args[0]
        assert 'UPDATE chapter_summaries' in call_args[0]
        params = call_args[1]
        assert isinstance(params[0], bytes)  # embedding BLOB
        assert params[1] == 'text-embedding-3-small'
        assert params[2] == 'ch-1'


def test_generate_single_embedding_no_summary():
    """Should return False when no summary exists"""
    gen = _make_mock_generator()

    with patch('src.utils.summaries.execute_query') as mock_query:
        mock_query.return_value = []

        from src.utils.summaries import generate_summary_embedding
        result = generate_summary_embedding('ch-nonexistent', gen)

        assert result is False
        gen.generate.assert_not_called()


def test_batch_generate_skips_existing():
    """batch_generate should only process summaries without embeddings"""
    with patch('src.utils.summaries.execute_query') as mock_query, \
         patch('src.utils.summaries.generate_summary_embedding') as mock_gen_single, \
         patch('src.utils.summaries.embedding_model_context') as mock_ctx:

        # First call: all summaries (4 total). Second call: missing only (2 need work).
        mock_query.side_effect = [
            [{'chapter_id': f'ch-{i}'} for i in range(4)],  # all summaries
            [{'chapter_id': 'ch-3'}, {'chapter_id': 'ch-4'}],  # missing embeddings
        ]
        mock_gen_single.return_value = True

        mock_generator = _make_mock_generator()
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_generator)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        from src.utils.summaries import batch_generate_summary_embeddings
        result = batch_generate_summary_embeddings(force=False)

        assert result['generated'] == 2
        assert result['skipped'] == 2  # 4 total - 2 needing work
        assert result['total'] == 4

        # Second query should filter for NULL embeddings
        second_query = mock_query.call_args_list[1][0][0]
        assert 'embedding IS NULL' in second_query


def test_batch_generate_force():
    """force=True should regenerate all, not just missing"""
    with patch('src.utils.summaries.execute_query') as mock_query, \
         patch('src.utils.summaries.generate_summary_embedding') as mock_gen_single, \
         patch('src.utils.summaries.embedding_model_context') as mock_ctx:

        # force=True only calls execute_query once (all summaries)
        mock_query.return_value = [
            {'chapter_id': 'ch-1'},
            {'chapter_id': 'ch-2'},
        ]
        mock_gen_single.return_value = True

        mock_generator = _make_mock_generator()
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_generator)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        from src.utils.summaries import batch_generate_summary_embeddings
        result = batch_generate_summary_embeddings(force=True)

        assert result['generated'] == 2
        assert result['skipped'] == 0

        # Only one query needed for force mode
        assert mock_query.call_count == 1
        query_used = mock_query.call_args[0][0]
        assert 'embedding IS NULL' not in query_used


# ─── Cache Tests ──────────────────────────────────────────────────────

def test_summary_cache_set_get():
    """Summary embeddings cache round-trip"""
    from src.utils.cache import LibraryCache

    cache = LibraryCache(enabled=True)
    matrix = np.random.randn(5, 384).astype(np.float32)
    metadata = [{'id': f'ch-{i}'} for i in range(5)]

    cache.set_summary_embeddings(matrix, metadata)
    result = cache.get_summary_embeddings()

    assert result is not None
    np.testing.assert_array_equal(result[0], matrix)
    assert result[1] == metadata


def test_summary_cache_invalidate():
    """Invalidation clears summary embeddings"""
    from src.utils.cache import LibraryCache

    cache = LibraryCache(enabled=True)
    matrix = np.random.randn(5, 384).astype(np.float32)
    metadata = [{'id': f'ch-{i}'} for i in range(5)]

    cache.set_summary_embeddings(matrix, metadata)
    assert cache.get_summary_embeddings() is not None

    cache.invalidate_summary_embeddings()
    assert cache.get_summary_embeddings() is None


def test_summary_cache_independent_of_chapter_embeddings():
    """Summary and chapter embeddings caches are independent"""
    from src.utils.cache import LibraryCache

    cache = LibraryCache(enabled=True)

    chap_matrix = np.random.randn(3, 384).astype(np.float32)
    chap_meta = [{'id': f'ch-{i}'} for i in range(3)]
    cache.set_embeddings(chap_matrix, chap_meta)

    sum_matrix = np.random.randn(2, 384).astype(np.float32)
    sum_meta = [{'id': f'sum-{i}'} for i in range(2)]
    cache.set_summary_embeddings(sum_matrix, sum_meta)

    # Invalidating chapter embeddings shouldn't affect summary embeddings
    cache.invalidate_embeddings()
    assert cache.get_embeddings() is None
    assert cache.get_summary_embeddings() is not None

    # And vice versa
    cache.set_embeddings(chap_matrix, chap_meta)
    cache.invalidate_summary_embeddings()
    assert cache.get_embeddings() is not None
    assert cache.get_summary_embeddings() is None


def test_clear_all_clears_summary_embeddings():
    """clear_all should clear summary embeddings too"""
    from src.utils.cache import LibraryCache

    cache = LibraryCache(enabled=True)
    matrix = np.random.randn(5, 384).astype(np.float32)
    cache.set_summary_embeddings(matrix, [{'id': f'ch-{i}'} for i in range(5)])

    cache.clear_all()
    assert cache.get_summary_embeddings() is None


def test_stats_include_summary_embeddings():
    """Cache stats should include summary embeddings info"""
    from src.utils.cache import LibraryCache

    cache = LibraryCache(enabled=True)
    matrix = np.random.randn(5, 384).astype(np.float32)
    cache.set_summary_embeddings(matrix, [{'id': f'ch-{i}'} for i in range(5)])

    stats = cache.stats()
    assert stats['summary_embeddings_loaded'] is True
    assert stats['summary_embeddings_count'] == 5
    assert stats['summary_embeddings_memory_mb'] > 0
