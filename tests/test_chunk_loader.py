"""Tests for chunk embedding loader."""

import io
import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np

from src.utils.chunk_loader import load_chunk_embeddings


class TestChunkLoader:
    def _setup_db(self):
        """Create a temp DB with books, chapters, and chunks tables."""
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(f.name)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute(
            "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, "
            "title TEXT, chapter_number INTEGER, file_path TEXT)"
        )
        conn.execute(
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, chapter_id TEXT, "
            "book_id TEXT, chunk_index INTEGER, content TEXT, "
            "word_count INTEGER, embedding BLOB, embedding_model TEXT, "
            "content_hash TEXT)"
        )
        return f.name, conn

    def _make_embedding_blob(self, dim=1536):
        arr = np.random.rand(dim).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue(), arr

    def test_loads_chunk_embeddings(self):
        db_path, conn = self._setup_db()
        conn.execute("INSERT INTO books VALUES ('b1', 'Test Book')")
        conn.execute(
            "INSERT INTO chapters VALUES ('ch1', 'b1', 'Chapter 1', 1, 'ch1.md')"
        )
        blob, arr = self._make_embedding_blob()
        conn.execute(
            "INSERT INTO chunks VALUES ('ch1:0', 'ch1', 'b1', 0, 'content', "
            "100, ?, 'text-embedding-3-small', 'hash1')",
            (blob,),
        )
        conn.commit()

        with patch("src.utils.chunk_loader.get_db_connection") as mock_db:
            mock_db.return_value.__enter__ = lambda s: conn
            mock_db.return_value.__exit__ = lambda s, *a: None
            matrix, metadata = load_chunk_embeddings(cache=None)

        assert matrix.shape == (1, 1536)
        assert metadata[0]["chunk_id"] == "ch1:0"
        assert metadata[0]["book_title"] == "Test Book"
        assert metadata[0]["chapter_title"] == "Chapter 1"
        conn.close()

    def test_returns_none_when_no_embeddings(self):
        db_path, conn = self._setup_db()
        conn.commit()

        with patch("src.utils.chunk_loader.get_db_connection") as mock_db:
            mock_db.return_value.__enter__ = lambda s: conn
            mock_db.return_value.__exit__ = lambda s, *a: None
            matrix, metadata = load_chunk_embeddings(cache=None)

        assert matrix is None
        assert metadata is None
        conn.close()
