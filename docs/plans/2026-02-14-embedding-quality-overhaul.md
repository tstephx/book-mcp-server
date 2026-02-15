<!-- project: book-mcp-server -->

# Embedding Quality Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve search quality by upgrading to OpenAI embeddings, adding sub-chapter chunking, and introducing Cohere reranking.

**Architecture:** Three-layer upgrade: (1) chunk chapters into ~500-word segments stored in a new `chunks` table, (2) embed chunks with OpenAI `text-embedding-3-small` (1536-dim) replacing local `all-MiniLM-L6-v2` (384-dim), (3) add Cohere `rerank-v3.5` as a post-retrieval reranker with graceful fallback. Search tools query chunks, map results back to chapters.

**Tech Stack:** OpenAI API (embeddings), Cohere API (reranking), SQLite (chunks table), numpy (vector ops), existing RRF/MMR fusion.

**Design doc:** `docs/plans/2026-02-14-embedding-quality-overhaul-design.md`

---

### Task 1: Chunker Module

**Files:**
- Create: `src/utils/chunker.py`
- Test: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
"""Tests for paragraph-grouping chunker."""

from src.utils.chunker import chunk_chapter


class TestChunkChapter:
    def test_short_chapter_single_chunk(self):
        """Chapters under 600 words become a single chunk."""
        text = "This is a short chapter. " * 20  # ~100 words
        chunks = chunk_chapter(text, target_words=500)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["content"] == text.strip()
        assert chunks[0]["word_count"] > 0

    def test_long_chapter_multiple_chunks(self):
        """Long chapters split into ~500-word chunks."""
        # Build a chapter with 5 clear paragraphs, each ~200 words
        paragraphs = []
        for i in range(5):
            paragraphs.append(f"Paragraph {i}. " + "word " * 199)
        text = "\n\n".join(paragraphs)  # ~1000 words total

        chunks = chunk_chapter(text, target_words=500)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk["word_count"] <= 700  # Some slack for paragraph boundaries

    def test_overlap_includes_last_paragraph(self):
        """Adjacent chunks share a boundary paragraph for context."""
        paragraphs = []
        for i in range(6):
            paragraphs.append(f"Topic {i}. " + "detail " * 149)
        text = "\n\n".join(paragraphs)  # 6 paragraphs x ~150 words = ~900 words

        chunks = chunk_chapter(text, target_words=400)
        assert len(chunks) >= 2
        # Last paragraph of chunk 0 should appear in chunk 1
        chunk_0_last_line = chunks[0]["content"].split("\n\n")[-1]
        assert chunk_0_last_line in chunks[1]["content"]

    def test_empty_input(self):
        """Empty or whitespace-only input returns empty list."""
        assert chunk_chapter("") == []
        assert chunk_chapter("   \n\n  ") == []

    def test_chunk_fields(self):
        """Each chunk dict has required fields."""
        text = "Hello world. " * 50
        chunks = chunk_chapter(text, target_words=500)
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "content" in chunk
            assert "word_count" in chunk
            assert isinstance(chunk["chunk_index"], int)
            assert isinstance(chunk["word_count"], int)
            assert len(chunk["content"]) > 0

    def test_no_empty_chunks(self):
        """Chunker never produces empty chunks."""
        text = "\n\n".join(["Short."] * 3 + ["Long paragraph. " * 100] + ["Short."] * 3)
        chunks = chunk_chapter(text, target_words=500)
        for chunk in chunks:
            assert chunk["word_count"] > 0
            assert chunk["content"].strip() != ""
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.chunker'`

**Step 3: Write minimal implementation**

```python
# src/utils/chunker.py
"""Paragraph-grouping chunker for sub-chapter retrieval.

Splits chapter text into ~500-word chunks along paragraph boundaries,
with one-paragraph overlap between adjacent chunks for context continuity.
"""

import re


def chunk_chapter(
    text: str,
    target_words: int = 500,
    min_chunk_words: int = 100,
) -> list[dict]:
    """Split chapter text into chunks along paragraph boundaries.

    Args:
        text: Full chapter text.
        target_words: Target words per chunk (~500).
        min_chunk_words: Minimum words for a standalone chunk.

    Returns:
        List of dicts with keys: chunk_index, content, word_count.
    """
    text = text.strip()
    if not text:
        return []

    total_words = len(text.split())

    # Short chapter threshold: 120% of target to avoid splitting into
    # one full chunk + one tiny chunk.
    if total_words <= int(target_words * 1.2):
        return [{"chunk_index": 0, "content": text, "word_count": total_words}]

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return [{"chunk_index": 0, "content": text, "word_count": total_words}]

    chunks = []
    current_paragraphs: list[str] = []
    current_words = 0
    prev_last_paragraph: str | None = None

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph would exceed target AND we already
        # have enough content, close the current chunk.
        if current_words + para_words > target_words and current_words >= min_chunk_words:
            chunks.append(_make_chunk(len(chunks), current_paragraphs))

            # Overlap: carry the last paragraph into the next chunk.
            prev_last_paragraph = current_paragraphs[-1]
            current_paragraphs = [prev_last_paragraph]
            current_words = len(prev_last_paragraph.split())

        current_paragraphs.append(para)
        current_words += para_words

    # Flush remaining content.
    if current_paragraphs:
        # If the leftover is too small, merge into previous chunk.
        if chunks and current_words < min_chunk_words:
            # Remove overlap paragraph to avoid double-counting.
            if prev_last_paragraph and current_paragraphs[0] == prev_last_paragraph:
                current_paragraphs = current_paragraphs[1:]
            if current_paragraphs:
                last = chunks[-1]
                merged_content = last["content"] + "\n\n" + "\n\n".join(current_paragraphs)
                chunks[-1] = _make_chunk(last["chunk_index"], None, merged_content)
        else:
            chunks.append(_make_chunk(len(chunks), current_paragraphs))

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newlines, filtering blanks."""
    raw = re.split(r"\n\n+", text)
    return [p.strip() for p in raw if p.strip()]


def _make_chunk(
    index: int,
    paragraphs: list[str] | None = None,
    content: str | None = None,
) -> dict:
    if content is None:
        content = "\n\n".join(paragraphs)
    return {
        "chunk_index": index,
        "content": content,
        "word_count": len(content.split()),
    }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/utils/chunker.py tests/test_chunker.py
git commit -m "feat: add paragraph-grouping chunker for sub-chapter retrieval"
```

---

### Task 2: OpenAI Embedding Wrapper

**Files:**
- Create: `src/utils/openai_embeddings.py`
- Test: `tests/test_openai_embeddings.py`

**Step 1: Write the failing tests**

```python
# tests/test_openai_embeddings.py
"""Tests for OpenAI embedding wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.utils.openai_embeddings import OpenAIEmbeddingGenerator


class TestOpenAIEmbeddingGenerator:
    def test_generate_single(self):
        """Generate embedding for a single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
            gen = OpenAIEmbeddingGenerator()
            result = gen.generate("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)

    def test_generate_batch(self):
        """Generate embeddings for multiple texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
            gen = OpenAIEmbeddingGenerator()
            result = gen.generate_batch(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1536)

    def test_generate_empty_raises(self):
        """Empty text raises ValueError."""
        import pytest

        gen = OpenAIEmbeddingGenerator.__new__(OpenAIEmbeddingGenerator)
        gen._client = MagicMock()

        with pytest.raises(ValueError):
            gen.generate("")

    def test_large_batch_splits(self):
        """Batches larger than max_batch_size are split into sub-batches."""
        embeddings = [[0.1] * 1536] * 5
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in embeddings]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            client = mock_openai.OpenAI.return_value
            client.embeddings.create.return_value = mock_response

            gen = OpenAIEmbeddingGenerator(max_batch_size=5)
            texts = [f"text {i}" for i in range(10)]
            result = gen.generate_batch(texts)

        # Should have been called twice (10 texts / 5 per batch)
        assert client.embeddings.create.call_count == 2
        assert result.shape == (10, 1536)

    def test_dimension_property(self):
        """Dimension reports 1536."""
        gen = OpenAIEmbeddingGenerator.__new__(OpenAIEmbeddingGenerator)
        assert gen.dimension == 1536
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_openai_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/utils/openai_embeddings.py
"""OpenAI embedding generation for semantic search.

Wraps the OpenAI embeddings API with batching support. Drop-in
replacement for the local EmbeddingGenerator interface.
"""

import logging
from typing import Optional

import numpy as np
import openai

logger = logging.getLogger(__name__)

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI text-embedding-3-small.

    Matches the interface of the local EmbeddingGenerator:
    generate(text) -> ndarray, generate_batch(texts) -> ndarray.
    """

    def __init__(self, max_batch_size: int = 2048):
        self._client = openai.OpenAI()
        self._max_batch_size = max_batch_size

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            1536-dimensional numpy array.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        response = self._client.embeddings.create(
            model=MODEL,
            input=[text],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_batch(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Ignored (uses max_batch_size from __init__).

        Returns:
            Array of shape (len(texts), 1536).
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty list")

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self._max_batch_size):
            batch = texts[i : i + self._max_batch_size]
            response = self._client.embeddings.create(
                model=MODEL,
                input=batch,
            )
            batch_embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        return DIMENSIONS
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_openai_embeddings.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/utils/openai_embeddings.py tests/test_openai_embeddings.py
git commit -m "feat: add OpenAI embedding wrapper (text-embedding-3-small)"
```

---

### Task 3: Cohere Reranker Module

**Files:**
- Create: `src/utils/reranker.py`
- Test: `tests/test_reranker.py`

**Step 1: Write the failing tests**

```python
# tests/test_reranker.py
"""Tests for Cohere reranker with graceful fallback."""

from unittest.mock import MagicMock, patch

from src.utils.reranker import rerank_results


class TestReranker:
    def _make_results(self, n: int) -> list[dict]:
        """Build dummy search results."""
        return [
            {
                "chapter_id": f"ch_{i}",
                "book_title": f"Book {i}",
                "chapter_title": f"Chapter {i}",
                "chunk_content": f"Content about topic {i}. " * 50,
                "rrf_score": 1.0 / (i + 1),
            }
            for i in range(n)
        ]

    def test_rerank_reorders_results(self):
        """Reranker reorders results based on Cohere scores."""
        results = self._make_results(3)

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=2, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.5),
            MagicMock(index=1, relevance_score=0.1),
        ]

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.return_value = mock_response
            reranked = rerank_results("test query", results, top_n=3)

        assert reranked[0]["chapter_id"] == "ch_2"
        assert reranked[1]["chapter_id"] == "ch_0"
        assert "rerank_score" in reranked[0]

    def test_fallback_on_api_error(self):
        """Returns original results if Cohere API fails."""
        results = self._make_results(3)

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.side_effect = Exception("API down")
            reranked = rerank_results("test query", results, top_n=3)

        assert [r["chapter_id"] for r in reranked] == ["ch_0", "ch_1", "ch_2"]
        assert "rerank_score" not in reranked[0]

    def test_empty_results(self):
        """Empty input returns empty output."""
        assert rerank_results("query", [], top_n=5) == []

    def test_top_n_limits_output(self):
        """top_n limits the number of returned results."""
        results = self._make_results(10)

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=i, relevance_score=1.0 - i * 0.1)
            for i in range(5)
        ]

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.return_value = mock_response
            reranked = rerank_results("query", results, top_n=5)

        assert len(reranked) == 5

    def test_disabled_returns_original(self):
        """rerank=False passes results through unchanged."""
        results = self._make_results(3)
        reranked = rerank_results("query", results, top_n=3, enabled=False)
        assert reranked == results
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/utils/reranker.py
"""Cohere reranker with graceful fallback.

Reranks search results using Cohere's cross-encoder for more precise
relevance scoring. Falls back to original ordering if the API is
unavailable.
"""

import logging
from typing import Optional

import cohere

logger = logging.getLogger(__name__)

MODEL = "rerank-v3.5"

# Module-level lazy client
_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2()
    return _client


def rerank_results(
    query: str,
    results: list[dict],
    top_n: int = 10,
    enabled: bool = True,
    content_key: str = "chunk_content",
) -> list[dict]:
    """Rerank search results using Cohere cross-encoder.

    Args:
        query: The search query.
        results: List of result dicts (must have content_key field).
        top_n: Number of results to return after reranking.
        enabled: Set False to skip reranking (pass-through).
        content_key: Dict key containing the text to rerank on.

    Returns:
        Reranked (or original) results, each with optional 'rerank_score'.
    """
    if not results or not enabled:
        return results[:top_n] if results else []

    documents = [r.get(content_key, "") for r in results]

    try:
        client = _get_client()
        response = client.rerank(
            model=MODEL,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked = []
        for item in response.results:
            result = dict(results[item.index])
            result["rerank_score"] = item.relevance_score
            reranked.append(result)

        return reranked

    except Exception as e:
        logger.warning(f"Cohere rerank failed, using original order: {e}")
        return results[:top_n]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: All 5 tests PASS

**Step 5: Install cohere dependency**

Run: `pip install cohere`

**Step 6: Commit**

```bash
git add src/utils/reranker.py tests/test_reranker.py
git commit -m "feat: add Cohere reranker with graceful fallback"
```

---

### Task 4: Chunks Table Schema Migration

**Files:**
- Modify: `agentic_pipeline/db/migrations.py` (append to MIGRATIONS and INDEXES lists)
- Modify: `src/database.py` (add ensure_chunks_table for MCP server startup)
- Test: `tests/test_chunks_migration.py`

**Step 1: Write the failing test**

```python
# tests/test_chunks_migration.py
"""Tests for chunks table migration."""

import sqlite3
import tempfile
from pathlib import Path

from agentic_pipeline.db.migrations import run_migrations


class TestChunksMigration:
    def test_chunks_table_created(self):
        """run_migrations creates the chunks table."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            )
            assert cursor.fetchone() is not None

    def test_chunks_table_columns(self):
        """chunks table has all expected columns."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(chunks)")
            columns = {row[1] for row in cursor.fetchall()}

            expected = {
                "id", "chapter_id", "book_id", "chunk_index",
                "content", "word_count", "embedding", "embedding_model",
                "content_hash", "created_at",
            }
            assert expected.issubset(columns)

    def test_chunks_indexes_created(self):
        """Chunk lookup indexes are created."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            run_migrations(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_chunks%'"
            )
            indexes = {row[0] for row in cursor.fetchall()}
            assert "idx_chunks_chapter" in indexes
            assert "idx_chunks_book" in indexes
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunks_migration.py -v`
Expected: FAIL — chunks table not found

**Step 3: Add migration**

In `agentic_pipeline/db/migrations.py`, append to the `MIGRATIONS` list (before the closing `]`):

```python
    # Chunks for sub-chapter retrieval (embedding quality overhaul)
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        chapter_id TEXT NOT NULL,
        book_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        word_count INTEGER NOT NULL,
        embedding BLOB,
        embedding_model TEXT,
        content_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chapter_id) REFERENCES chapters(id)
    )
    """,
```

Append to the `INDEXES` list:

```python
    "CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id)",
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunks_migration.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/db/migrations.py tests/test_chunks_migration.py
git commit -m "feat: add chunks table schema migration"
```

---

### Task 5: Chunk Loader (replaces chapter embedding loader)

**Files:**
- Create: `src/utils/chunk_loader.py`
- Test: `tests/test_chunk_loader.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunk_loader.py
"""Tests for chunk embedding loader."""

import io
import sqlite3
import tempfile
from pathlib import Path
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunk_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/utils/chunk_loader.py
"""Load chunk embeddings from DB or cache for search.

Replaces embedding_loader.py's chapter-level loading with chunk-level.
The old load_chapter_embeddings() still exists for backward compat.
"""

import io
import logging
from typing import Optional

import numpy as np

from ..database import get_db_connection
from .cache import get_cache

logger = logging.getLogger(__name__)


def load_chunk_embeddings(
    cache=None,
) -> tuple[Optional[np.ndarray], Optional[list[dict]]]:
    """Load chunk embeddings from cache or database.

    Args:
        cache: LibraryCache instance (uses global cache if None).

    Returns:
        (embeddings_matrix, chunk_metadata) or (None, None).
        Metadata dicts: chunk_id, chapter_id, book_id, book_title,
        chapter_title, chapter_number, chunk_index, content, file_path.
    """
    if cache is None:
        cache = get_cache()

    if cache is not None:
        cached = cache.get_embeddings()
        if cached:
            return cached

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                k.id AS chunk_id,
                k.chapter_id,
                k.book_id,
                k.chunk_index,
                k.content,
                k.embedding,
                c.title AS chapter_title,
                c.chapter_number,
                c.file_path,
                b.title AS book_title
            FROM chunks k
            JOIN chapters c ON k.chapter_id = c.id
            JOIN books b ON k.book_id = b.id
            WHERE k.embedding IS NOT NULL
            ORDER BY k.id
        """)
        rows = cursor.fetchall()

    if not rows:
        logger.warning("No chunk embeddings found in database")
        return None, None

    embeddings = []
    metadata = []

    for row in rows:
        embedding = np.load(io.BytesIO(row["embedding"]))
        embeddings.append(embedding)
        metadata.append({
            "chunk_id": row["chunk_id"],
            "chapter_id": row["chapter_id"],
            "book_id": row["book_id"],
            "book_title": row["book_title"],
            "chapter_title": row["chapter_title"],
            "chapter_number": row["chapter_number"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "file_path": row["file_path"],
        })

    matrix = np.vstack(embeddings)

    if cache is not None:
        cache.set_embeddings(matrix, metadata)

    logger.info(f"Loaded {len(metadata)} chunk embeddings from DB")
    return matrix, metadata
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunk_loader.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/utils/chunk_loader.py tests/test_chunk_loader.py
git commit -m "feat: add chunk embedding loader for search"
```

---

### Task 6: Update Semantic Search Tool

**Files:**
- Modify: `src/tools/semantic_search_tool.py`
- Test: `tests/test_semantic_search_updated.py`

This task rewires `semantic_search` to:
1. Use `OpenAIEmbeddingGenerator` for query embedding
2. Use `load_chunk_embeddings()` instead of `load_chapter_embeddings()`
3. Return chunk content as the excerpt (no runtime excerpt extraction needed)
4. Add `rerank: bool = True` parameter
5. Call `rerank_results()` before returning

**Step 1: Write the failing test**

```python
# tests/test_semantic_search_updated.py
"""Tests that semantic_search uses chunks and reranking."""

from unittest.mock import MagicMock, patch

import numpy as np


def test_semantic_search_uses_chunks():
    """semantic_search should call load_chunk_embeddings, not load_chapter_embeddings."""
    with patch("src.tools.semantic_search_tool.load_chunk_embeddings") as mock_chunks, \
         patch("src.tools.semantic_search_tool.OpenAIEmbeddingGenerator") as mock_gen_cls, \
         patch("src.tools.semantic_search_tool.rerank_results") as mock_rerank, \
         patch("src.tools.semantic_search_tool.find_top_k") as mock_topk:

        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.zeros(1536)
        mock_gen_cls.return_value = mock_gen

        mock_chunks.return_value = (
            np.zeros((2, 1536)),
            [
                {"chunk_id": "c1:0", "book_id": "b1", "book_title": "Book",
                 "chapter_title": "Ch1", "chapter_number": 1, "content": "text",
                 "chunk_index": 0, "chapter_id": "c1", "file_path": "c1.md"},
                {"chunk_id": "c2:0", "book_id": "b2", "book_title": "Book2",
                 "chapter_title": "Ch2", "chapter_number": 2, "content": "other",
                 "chunk_index": 0, "chapter_id": "c2", "file_path": "c2.md"},
            ],
        )
        mock_topk.return_value = [(0, 0.9)]
        mock_rerank.return_value = [
            {"chunk_id": "c1:0", "book_title": "Book", "chapter_title": "Ch1",
             "chapter_number": 1, "content": "text", "rerank_score": 0.95,
             "chunk_content": "text"},
        ]

        # Import the registration function and create a mock MCP
        from src.tools.semantic_search_tool import register_semantic_search_tools
        mcp = MagicMock()
        captured = {}

        def capture_tool():
            def decorator(fn):
                captured["semantic_search"] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_semantic_search_tools(mcp)

        result = captured["semantic_search"]("test query", limit=5)

        mock_chunks.assert_called_once()
        assert result["results"][0]["excerpt"] == "text"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_search_updated.py -v`
Expected: FAIL — old imports still in use

**Step 3: Update `src/tools/semantic_search_tool.py`**

Replace the entire file contents. Key changes:
- Import `OpenAIEmbeddingGenerator` instead of `embedding_model_context`
- Import `load_chunk_embeddings` instead of `load_chapter_embeddings`
- Import `rerank_results`
- Remove excerpt extraction (chunk content IS the excerpt)
- Add `rerank` parameter

```python
# src/tools/semantic_search_tool.py
"""Semantic search tool — queries chunk embeddings with optional reranking."""

import logging
from typing import Optional

import numpy as np

from ..utils.openai_embeddings import OpenAIEmbeddingGenerator
from ..utils.chunk_loader import load_chunk_embeddings
from ..utils.reranker import rerank_results
from ..utils.vector_store import find_top_k
from ..utils.cache import get_cache
from ..schemas.tool_schemas import SemanticSearchInput

logger = logging.getLogger(__name__)

# Module-level lazy generator (avoid re-creating per call)
_generator: Optional[OpenAIEmbeddingGenerator] = None


def _get_generator() -> OpenAIEmbeddingGenerator:
    global _generator
    if _generator is None:
        _generator = OpenAIEmbeddingGenerator()
    return _generator


def register_semantic_search_tools(mcp):
    """Register semantic search tool with MCP server."""

    @mcp.tool()
    def semantic_search(
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        rerank: bool = True,
    ) -> dict:
        """Search books using semantic similarity

        Uses embeddings to find conceptually similar content, even if
        exact keywords don't match. Results are reranked for precision.

        Args:
            query: Search query (e.g., "container networking concepts")
            limit: Maximum results to return (1-20, default: 5)
            min_similarity: Minimum similarity score 0.0-1.0 (default: 0.3)
            rerank: Rerank results with Cohere for better precision (default: True)

        Returns:
            Dictionary with results containing book_title, chapter_title,
            chapter_number, similarity, excerpt.
        """
        try:
            try:
                validated = SemanticSearchInput(
                    query=query, limit=limit, min_similarity=min_similarity
                )
            except Exception as e:
                return {"error": f"Invalid input: {e}", "results": []}

            generator = _get_generator()
            query_embedding = generator.generate(validated.query)

            embeddings_matrix, chunk_metadata = load_chunk_embeddings()

            if embeddings_matrix is None:
                return {
                    "message": "No chunk embeddings found. Run embed-library first.",
                    "results": [],
                }

            # Over-fetch for reranking
            fetch_k = validated.limit * 3 if rerank else validated.limit
            top_results = find_top_k(
                query_embedding,
                embeddings_matrix,
                k=fetch_k,
                min_similarity=validated.min_similarity,
            )

            # Build candidate list
            candidates = []
            for idx, similarity in top_results:
                meta = chunk_metadata[idx]
                candidates.append({
                    "chapter_id": meta["chapter_id"],
                    "chunk_id": meta["chunk_id"],
                    "book_id": meta["book_id"],
                    "book_title": meta["book_title"],
                    "chapter_title": meta["chapter_title"],
                    "chapter_number": meta["chapter_number"],
                    "similarity": round(similarity, 3),
                    "chunk_content": meta["content"],
                })

            # Rerank
            if rerank and candidates:
                candidates = rerank_results(
                    validated.query,
                    candidates,
                    top_n=validated.limit,
                    content_key="chunk_content",
                )
            else:
                candidates = candidates[: validated.limit]

            # Format output
            results = []
            for r in candidates:
                results.append({
                    "book_title": r["book_title"],
                    "chapter_title": r["chapter_title"],
                    "chapter_number": r["chapter_number"],
                    "similarity": r.get("similarity", 0),
                    "rerank_score": r.get("rerank_score"),
                    "excerpt": r["chunk_content"][:500],
                })

            return {
                "query": validated.query,
                "results": results,
                "total_found": len(results),
            }

        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    @mcp.resource("book://semantic-context/{query}")
    async def semantic_context(query: str) -> str:
        """Provide semantic context for RAG pattern."""
        try:
            search_results = semantic_search(query, limit=3, min_similarity=0.4)
            if "error" in search_results:
                return f"Error retrieving context: {search_results['error']}"
            results = search_results.get("results", [])
            if not results:
                return f"No relevant context found for: {query}"
            parts = [f"Relevant context for '{query}':\n"]
            for i, r in enumerate(results, 1):
                parts.append(
                    f"\n[{i}] From '{r['book_title']}' - "
                    f"Chapter {r['chapter_number']}: {r['chapter_title']}\n"
                    f"(Similarity: {r['similarity']:.2f})\n"
                    f"{r['excerpt']}\n"
                )
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Semantic context error: {e}", exc_info=True)
            return f"Error: {e}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_search_updated.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tools/semantic_search_tool.py tests/test_semantic_search_updated.py
git commit -m "feat: rewire semantic search to use chunks + reranking"
```

---

### Task 7: Update Hybrid Search Tool

**Files:**
- Modify: `src/tools/hybrid_search_tool.py`
- Test: `tests/test_hybrid_search_updated.py`

Same pattern as Task 6: switch to chunks, add reranking, use OpenAI embeddings for query.

**Step 1: Write the failing test**

```python
# tests/test_hybrid_search_updated.py
"""Tests that hybrid_search uses chunks and reranking."""

from unittest.mock import MagicMock, patch

import numpy as np


def test_hybrid_search_uses_chunks():
    """hybrid_search should call load_chunk_embeddings."""
    with patch("src.tools.hybrid_search_tool.load_chunk_embeddings") as mock_chunks, \
         patch("src.tools.hybrid_search_tool.OpenAIEmbeddingGenerator") as mock_gen_cls, \
         patch("src.tools.hybrid_search_tool.rerank_results") as mock_rerank, \
         patch("src.tools.hybrid_search_tool.find_top_k") as mock_topk, \
         patch("src.tools.hybrid_search_tool.full_text_search") as mock_fts, \
         patch("src.tools.hybrid_search_tool.reciprocal_rank_fusion") as mock_rrf:

        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.zeros(1536)
        mock_gen_cls.return_value = mock_gen

        mock_chunks.return_value = (
            np.zeros((1, 1536)),
            [{"chunk_id": "c1:0", "id": "c1:0", "book_id": "b1", "book_title": "Book",
              "chapter_title": "Ch1", "chapter_number": 1, "content": "text",
              "chunk_index": 0, "chapter_id": "c1", "file_path": "c1.md"}],
        )
        mock_topk.return_value = [(0, 0.9)]
        mock_fts.return_value = {"results": []}
        mock_rrf.return_value = [
            {"chapter_id": "c1", "book_id": "b1", "book_title": "Book",
             "chapter_title": "Ch1", "chapter_number": 1,
             "rrf_score": 0.5, "sources": ["semantic"]},
        ]
        mock_rerank.return_value = [
            {"chapter_id": "c1", "book_title": "Book", "chapter_title": "Ch1",
             "chapter_number": 1, "rrf_score": 0.5, "sources": ["semantic"],
             "chunk_content": "text", "rerank_score": 0.9},
        ]

        from src.tools.hybrid_search_tool import register_hybrid_search_tools
        mcp = MagicMock()
        captured = {}

        def capture_tool():
            def decorator(fn):
                captured["hybrid_search"] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_hybrid_search_tools(mcp)

        result = captured["hybrid_search"]("test query", limit=5)

        mock_chunks.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_search_updated.py -v`
Expected: FAIL — old imports

**Step 3: Update `src/tools/hybrid_search_tool.py`**

Key changes:
- Import `OpenAIEmbeddingGenerator` instead of `embedding_model_context`
- Import `load_chunk_embeddings` instead of `load_chapter_embeddings`
- Import `rerank_results`
- Add `rerank` parameter
- Remove runtime excerpt extraction — use chunk content
- Chunk metadata includes `content` field, used for both reranking input and excerpt output

The full updated file follows the same pattern as the semantic search update. The RRF fusion and MMR logic stay the same — they operate on result dicts. The main change is the data source (chunks) and the reranking step after fusion.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_search_updated.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tools/hybrid_search_tool.py tests/test_hybrid_search_updated.py
git commit -m "feat: rewire hybrid search to use chunks + reranking"
```

---

### Task 8: Update Processing Adapter for New Embedding Pipeline

**Files:**
- Modify: `agentic_pipeline/adapters/processing_adapter.py`
- Test: `tests/test_processing_adapter_chunks.py`

After `process_book()` writes chapters, the EMBEDDING state now:
1. Chunks the chapters using `chunk_chapter()`
2. Inserts chunk rows into the `chunks` table
3. Generates embeddings via OpenAI (not local model)
4. Also updates `chapters.embedding` with OpenAI for backward compat

**Step 1: Write the failing test**

```python
# tests/test_processing_adapter_chunks.py
"""Tests that the adapter generates chunks and OpenAI embeddings."""

import io
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _setup_db_with_chapters():
    """Create a temp DB with a book and chapters."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, title TEXT, "
        "chapter_number INTEGER, file_path TEXT, word_count INTEGER, "
        "embedding BLOB, embedding_model TEXT, content_hash TEXT, "
        "file_mtime REAL, embedding_updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, chapter_id TEXT, book_id TEXT, "
        "chunk_index INTEGER, content TEXT, word_count INTEGER, "
        "embedding BLOB, embedding_model TEXT, content_hash TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO books VALUES ('b1', 'Test Book')")
    conn.execute(
        "INSERT INTO chapters VALUES ('ch1', 'b1', 'Chapter 1', 1, 'ch1.md', "
        "500, NULL, NULL, NULL, NULL, NULL)"
    )
    conn.commit()
    return f.name, conn


def test_generate_embeddings_creates_chunks():
    """generate_embeddings should chunk chapters and store in chunks table."""
    db_path, conn = _setup_db_with_chapters()

    chapter_content = "Paragraph one about testing. " * 100 + "\n\n" + "Paragraph two about quality. " * 100

    with patch("agentic_pipeline.adapters.processing_adapter.get_pipeline_db") as mock_db, \
         patch("agentic_pipeline.adapters.processing_adapter.OpenAIEmbeddingGenerator") as mock_gen_cls, \
         patch.object(Path, "read_text", return_value=chapter_content):

        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = lambda s, *a: None

        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = np.random.rand(2, 1536).astype(np.float32)
        mock_gen_cls.return_value = mock_gen

        from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter
        adapter = ProcessingAdapter.__new__(ProcessingAdapter)
        adapter.db_path = Path(db_path)
        adapter._embedding_generator = None

        result = adapter.generate_embeddings(book_id="b1")

    assert result.success
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = 'b1'")
    chunk_count = cursor.fetchone()[0]
    assert chunk_count > 0

    conn.close()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_processing_adapter_chunks.py -v`
Expected: FAIL — adapter still uses local embedding model

**Step 3: Update `agentic_pipeline/adapters/processing_adapter.py`**

Key changes to `generate_embeddings()`:
1. Import `chunk_chapter` from `src.utils.chunker`
2. Import `OpenAIEmbeddingGenerator` from `src.utils.openai_embeddings`
3. For each chapter: read content → `chunk_chapter()` → insert chunk rows
4. Batch all chunk texts → `OpenAIEmbeddingGenerator.generate_batch()` → update chunk embeddings
5. Remove the old local `EmbeddingGenerator` import

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_processing_adapter_chunks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/adapters/processing_adapter.py tests/test_processing_adapter_chunks.py
git commit -m "feat: processing adapter generates chunks + OpenAI embeddings"
```

---

### Task 9: CLI Commands for Library Migration

**Files:**
- Modify: `agentic_pipeline/cli.py` (add `chunk-library` and `embed-library` commands)
- Test: `tests/test_cli_migration_commands.py`

These commands are the one-time migration tools to chunk and re-embed the existing 195-book library.

**Step 1: Write the failing test**

```python
# tests/test_cli_migration_commands.py
"""Tests for chunk-library and embed-library CLI commands."""

from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_chunk_library_dry_run():
    """chunk-library --dry-run shows what would be chunked."""
    from agentic_pipeline.cli import main

    runner = CliRunner()
    with patch("agentic_pipeline.cli.get_db_path") as mock_path, \
         patch("agentic_pipeline.cli.chunk_all_books") as mock_chunk:
        mock_path.return_value = "/tmp/test.db"
        mock_chunk.return_value = {"books": 5, "chapters": 50, "chunks_created": 200}

        result = runner.invoke(main, ["chunk-library", "--dry-run"])

    assert result.exit_code == 0
    mock_chunk.assert_called_once()


def test_embed_library_dry_run():
    """embed-library --dry-run shows what would be embedded."""
    from agentic_pipeline.cli import main

    runner = CliRunner()
    with patch("agentic_pipeline.cli.get_db_path") as mock_path, \
         patch("agentic_pipeline.cli.embed_all_chunks") as mock_embed:
        mock_path.return_value = "/tmp/test.db"
        mock_embed.return_value = {"chunks_embedded": 0, "total_chunks": 200}

        result = runner.invoke(main, ["embed-library", "--dry-run"])

    assert result.exit_code == 0
    mock_embed.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_migration_commands.py -v`
Expected: FAIL — commands don't exist

**Step 3: Add CLI commands to `agentic_pipeline/cli.py`**

Add two new Click commands:

```python
@main.command("chunk-library")
@click.option("--dry-run", is_flag=True, help="Preview without making changes")
def chunk_library_cmd(dry_run: bool):
    """Generate chunks for all books in the library."""
    from agentic_pipeline.db.config import get_db_path
    db_path = str(get_db_path())
    result = chunk_all_books(db_path, dry_run=dry_run)
    # Display with Rich table...


@main.command("embed-library")
@click.option("--dry-run", is_flag=True, help="Preview without making changes")
@click.option("--batch-size", default=100, help="Chunks per API call")
def embed_library_cmd(dry_run: bool, batch_size: int):
    """Generate OpenAI embeddings for all chunks."""
    from agentic_pipeline.db.config import get_db_path
    db_path = str(get_db_path())
    result = embed_all_chunks(db_path, dry_run=dry_run, batch_size=batch_size)
    # Display with Rich table...
```

The `chunk_all_books()` and `embed_all_chunks()` functions should live in a new `agentic_pipeline/library/migration.py` module that:
- `chunk_all_books`: reads all chapters, runs `chunk_chapter()`, inserts into `chunks` table
- `embed_all_chunks`: reads all chunks without embeddings, batches through `OpenAIEmbeddingGenerator`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_migration_commands.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/cli.py agentic_pipeline/library/migration.py tests/test_cli_migration_commands.py
git commit -m "feat: add chunk-library and embed-library CLI commands"
```

---

### Task 10: Integration Test — Full Search Pipeline

**Files:**
- Create: `tests/test_search_integration.py`

End-to-end test with a real temp DB: insert a book, chapters, chunk them, embed them (mocked), search, verify results map correctly.

**Step 1: Write the integration test**

```python
# tests/test_search_integration.py
"""Integration test: chunk → embed → search → rerank pipeline."""

import io
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.utils.chunker import chunk_chapter


def test_full_search_pipeline():
    """Insert book → chunk → embed → semantic search returns correct results."""
    # Setup DB
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
```

**Step 2: Run integration test**

Run: `python -m pytest tests/test_search_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_search_integration.py
git commit -m "test: add integration test for chunk → search pipeline"
```

---

### Task 11: Run Full Migration on Library

This is the production run — not a coding task but an operational one.

**Step 1: Run chunk generation**

```bash
source .venv/bin/activate
agentic-pipeline chunk-library --dry-run   # Preview first
agentic-pipeline chunk-library             # Execute
```

Expected: ~10,000-15,000 chunks created from ~3,400 chapters.

**Step 2: Run embedding generation**

```bash
agentic-pipeline embed-library --dry-run   # Preview (shows cost estimate)
agentic-pipeline embed-library             # Execute (~$0.21 cost)
```

Expected: All chunks embedded with `text-embedding-3-small`.

**Step 3: Verify search works**

Test a few queries manually in Claude Desktop to verify results are returning and reranking is working.

**Step 4: Commit any config changes**

```bash
git add -A && git commit -m "chore: complete library migration to chunk embeddings"
```

---

### Task 12: Run Full Test Suite

**Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All existing tests PASS (plus the new ones from Tasks 1-10). Fix any regressions from the import changes in the search tools.

**Step 2: Commit any fixes**

```bash
git add -A && git commit -m "fix: resolve test regressions from embedding overhaul"
```
