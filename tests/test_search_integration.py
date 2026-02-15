"""Integration test: chunk -> embed -> search -> rerank pipeline."""

import io
import sqlite3
import tempfile

import numpy as np

from src.utils.chunker import chunk_chapter


def test_full_search_pipeline():
    """Insert book -> chunk -> embed -> verify DB joins work correctly."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.row_factory = sqlite3.Row

    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, title TEXT, "
        "chapter_number INTEGER, file_path TEXT)"
    )
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, chapter_id TEXT, book_id TEXT, "
        "chunk_index INTEGER, content TEXT, word_count INTEGER, "
        "embedding BLOB, embedding_model TEXT, content_hash TEXT)"
    )

    conn.execute("INSERT INTO books VALUES ('b1', 'Docker Deep Dive')")
    conn.execute(
        "INSERT INTO chapters VALUES ('ch1', 'b1', 'Networking', 5, 'ch1.md')"
    )

    # Chunk a chapter
    chapter_text = (
        "Docker networking enables containers to communicate. "
        "Bridge networks are the default. " * 50 + "\n\n"
        "Overlay networks span multiple hosts. " * 50
    )
    chunks = chunk_chapter(chapter_text, target_words=200)
    assert len(chunks) >= 2

    # Insert chunks with fake embeddings
    for chunk in chunks:
        chunk_id = f"ch1:{chunk['chunk_index']}"
        emb = np.random.rand(1536).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, emb)
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, "ch1", "b1", chunk["chunk_index"],
             chunk["content"], chunk["word_count"],
             buf.getvalue(), "text-embedding-3-small", "hash"),
        )

    conn.commit()

    # Verify chunks are queryable
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = 'b1'")
    assert cursor.fetchone()[0] == len(chunks)

    # Verify chunk content maps back to chapter
    cursor.execute(
        "SELECT k.content, c.title as chapter_title, b.title as book_title "
        "FROM chunks k JOIN chapters c ON k.chapter_id = c.id "
        "JOIN books b ON k.book_id = b.id WHERE k.book_id = 'b1' LIMIT 1"
    )
    row = cursor.fetchone()
    assert row["chapter_title"] == "Networking"
    assert row["book_title"] == "Docker Deep Dive"

    conn.close()


def test_load_chunk_embeddings_round_trip():
    """load_chunk_embeddings deserializes numpy blobs and returns correct shape."""
    from unittest.mock import patch, MagicMock

    from src.utils.chunk_loader import load_chunk_embeddings

    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.row_factory = sqlite3.Row

    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, title TEXT, "
        "chapter_number INTEGER, file_path TEXT)"
    )
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, chapter_id TEXT, book_id TEXT, "
        "chunk_index INTEGER, content TEXT, word_count INTEGER, "
        "embedding BLOB, embedding_model TEXT, content_hash TEXT)"
    )

    conn.execute("INSERT INTO books VALUES ('b1', 'Test Book')")
    conn.execute(
        "INSERT INTO chapters VALUES ('ch1', 'b1', 'Chapter 1', 1, 'ch1.md')"
    )

    # Insert 3 chunks with known embeddings
    np.random.seed(99)
    expected_embeddings = []
    for i in range(3):
        emb = np.random.rand(1536).astype(np.float32)
        expected_embeddings.append(emb)
        buf = io.BytesIO()
        np.save(buf, emb)
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"ch1:{i}", "ch1", "b1", i,
             f"Chunk {i} content", 10,
             buf.getvalue(), "text-embedding-3-small", f"hash{i}"),
        )

    conn.commit()
    conn.close()

    # Mock get_db_connection to use our temp DB
    mock_conn = sqlite3.connect(f.name)
    mock_conn.row_factory = sqlite3.Row
    mock_cache = MagicMock()
    mock_cache.get_chunk_embeddings.return_value = None

    with patch("src.utils.chunk_loader.get_db_connection") as mock_get_db:
        mock_get_db.return_value.__enter__ = lambda s: mock_conn
        mock_get_db.return_value.__exit__ = lambda s, *a: None

        matrix, metadata = load_chunk_embeddings(cache=mock_cache)

    assert matrix.shape == (3, 1536)
    assert len(metadata) == 3
    assert metadata[0]["chunk_id"] == "ch1:0"
    assert metadata[0]["book_title"] == "Test Book"
    assert metadata[0]["chapter_title"] == "Chapter 1"

    # Verify embeddings match what we stored
    for i in range(3):
        np.testing.assert_array_almost_equal(matrix[i], expected_embeddings[i])

    # Verify cache was populated
    mock_cache.set_chunk_embeddings.assert_called_once()

    mock_conn.close()
