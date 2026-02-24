"""Tests for ProcessingAdapter.generate_embeddings() chunking + OpenAI embedding."""

import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_pipeline.db.migrations import run_migrations

# Build mock book_ingestion modules.  These are installed into sys.modules
# before importing processing_adapter so that the module-level
# "from book_ingestion import ..." resolves to our mocks.
#
# We use assignment (not setdefault) so we win even when
# test_processing_adapter.py was collected first and already populated
# sys.modules with the real book_ingestion package.
#
# Critically we do NOT reload agentic_pipeline.adapters.processing_adapter —
# reloading would create new class objects (ProcessingResult, etc.) and break
# isinstance() checks in test_processing_adapter.py.  Instead we patch the
# already-imported module's attributes directly so @patch decorators can find
# and replace the correct names.
_mock_bi = ModuleType("book_ingestion")
_mock_bi.BookIngestionApp = MagicMock()
_mock_bi.ProcessingMode = MagicMock()
_mock_bi.PipelineResult = MagicMock()
_mock_bi_ports = ModuleType("book_ingestion.ports")
_mock_bi_ports_llm = ModuleType("book_ingestion.ports.llm_fallback")
_mock_bi_ports_llm.LLMFallbackPort = MagicMock()
_mock_bi_ports_llm.LLMFallbackRequest = MagicMock()
_mock_bi_ports_llm.LLMFallbackResponse = MagicMock()
_mock_bi.ports = _mock_bi_ports

_BI_MODULE_NAMES = [
    "book_ingestion",
    "book_ingestion.ports",
    "book_ingestion.ports.llm_fallback",
]
_BI_MOCKS = [_mock_bi, _mock_bi_ports, _mock_bi_ports_llm]
_original_bi_modules: dict = {}

for _mod_name, _mod in zip(_BI_MODULE_NAMES, _BI_MOCKS):
    _original_bi_modules[_mod_name] = sys.modules.get(_mod_name)
    sys.modules[_mod_name] = _mod

# Import (or retrieve already-cached) processing_adapter without reloading it.
# The @patch decorators in each test target names inside this module's
# namespace, which is sufficient — the module was already imported with
# whatever book_ingestion was present, but chunk_chapter and
# OpenAIEmbeddingGenerator come from src.utils.* and are correctly patched.
import agentic_pipeline.adapters.processing_adapter  # noqa: F401, E402

from agentic_pipeline.adapters.processing_adapter import (  # noqa: E402
    ProcessingAdapter,
    EmbeddingResult,
)


@pytest.fixture(autouse=True, scope="module")
def _restore_bi_modules():
    """After all tests in this module run, restore sys.modules to original state."""
    yield
    for mod_name in _BI_MODULE_NAMES:
        original = _original_bi_modules.get(mod_name)
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


def _create_test_db(tmp_path: Path) -> Path:
    """Create a test database with library + pipeline tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE books (
        id TEXT PRIMARY KEY, title TEXT, author TEXT
    )""")
    conn.execute("""CREATE TABLE chapters (
        id TEXT PRIMARY KEY, book_id TEXT, title TEXT,
        chapter_number INTEGER, file_path TEXT
    )""")
    conn.commit()
    conn.close()
    # Run pipeline migrations (creates chunks table)
    run_migrations(db_path)
    return db_path


def _insert_book_and_chapter(db_path: Path, book_id: str, chapter_id: str,
                              file_path: str):
    """Insert a book and chapter into the test DB."""
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO books (id, title, author) VALUES (?, ?, ?)",
                 (book_id, "Test Book", "Author"))
    conn.execute(
        "INSERT INTO chapters (id, book_id, title, chapter_number, file_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (chapter_id, book_id, "Chapter 1", 1, file_path),
    )
    conn.commit()
    conn.close()


class TestGenerateEmbeddings:
    """Tests for the chunking + embedding pipeline in ProcessingAdapter."""

    @patch("agentic_pipeline.adapters.processing_adapter.chunk_chapter")
    @patch("agentic_pipeline.adapters.processing_adapter.OpenAIEmbeddingGenerator")
    def test_chunks_and_embeds_chapter(self, mock_emb_cls, mock_chunk, tmp_path):
        """generate_embeddings creates chunks and embeds them."""
        db_path = _create_test_db(tmp_path)

        # Create a chapter file
        books_dir = db_path.parent / "books"
        books_dir.mkdir()
        ch_file = books_dir / "ch1.md"
        ch_file.write_text("Some chapter content here.")

        _insert_book_and_chapter(db_path, "b1", "c1", str(ch_file))

        # Mock chunk_chapter to return 2 chunks
        mock_chunk.return_value = [
            {"chunk_index": 0, "content": "chunk zero", "word_count": 50},
            {"chunk_index": 1, "content": "chunk one", "word_count": 45},
        ]

        # Mock embedding generator
        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = np.zeros((2, 1536))
        mock_emb_cls.return_value = mock_gen

        adapter = ProcessingAdapter(db_path=db_path)
        result = adapter.generate_embeddings(book_id="b1")

        assert result.success is True
        assert result.chapters_processed == 1
        assert result.chunks_created == 2
        assert result.chunks_embedded == 2

        # Verify chunks are in DB
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM chunks ORDER BY chunk_index").fetchall()
        assert len(rows) == 2
        assert rows[0]["id"] == "c1:0"
        assert rows[1]["id"] == "c1:1"
        assert rows[0]["content"] == "chunk zero"
        # Verify embeddings were stored
        assert rows[0]["embedding"] is not None
        assert rows[0]["embedding_model"] == "text-embedding-3-large"
        conn.close()

    @patch("agentic_pipeline.adapters.processing_adapter.chunk_chapter")
    @patch("agentic_pipeline.adapters.processing_adapter.OpenAIEmbeddingGenerator")
    def test_skips_already_chunked_chapters(self, mock_emb_cls, mock_chunk, tmp_path):
        """generate_embeddings skips chapters that already have chunks."""
        db_path = _create_test_db(tmp_path)

        books_dir = db_path.parent / "books"
        books_dir.mkdir()
        ch_file = books_dir / "ch1.md"
        ch_file.write_text("Some content.")

        _insert_book_and_chapter(db_path, "b1", "c1", str(ch_file))

        # Pre-insert a chunk so chapter is already processed
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, word_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("c1:0", "c1", "b1", 0, "existing chunk", 20),
        )
        conn.commit()
        conn.close()

        mock_emb_cls.return_value = MagicMock()

        adapter = ProcessingAdapter(db_path=db_path)
        result = adapter.generate_embeddings(book_id="b1")

        assert result.success is True
        assert result.chapters_processed == 0
        mock_chunk.assert_not_called()

    @patch("agentic_pipeline.adapters.processing_adapter.OpenAIEmbeddingGenerator")
    def test_no_chapters_returns_success(self, mock_emb_cls, tmp_path):
        """generate_embeddings returns success with 0 processed when no chapters."""
        db_path = _create_test_db(tmp_path)
        mock_emb_cls.return_value = MagicMock()

        adapter = ProcessingAdapter(db_path=db_path)
        result = adapter.generate_embeddings()

        assert result.success is True
        assert result.chapters_processed == 0
        assert result.chunks_created == 0
