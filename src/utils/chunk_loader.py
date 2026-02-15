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
        cached = cache.get_chunk_embeddings()
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
                c.word_count,
                b.title AS book_title,
                b.author
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
            "author": row["author"],
            "chapter_title": row["chapter_title"],
            "chapter_number": row["chapter_number"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "file_path": row["file_path"],
            "word_count": row["word_count"],
        })

    matrix = np.vstack(embeddings)

    if cache is not None:
        cache.set_chunk_embeddings(matrix, metadata)

    logger.info(f"Loaded {len(metadata)} chunk embeddings from DB")
    return matrix, metadata


def best_chunk_per_chapter(
    chunk_results: list[dict],
) -> list[dict]:
    """Aggregate chunk-level search results to chapter level.

    Takes the best-scoring chunk per chapter and returns chapter-level
    results with the best chunk's content attached as ``excerpt``.

    Each item in *chunk_results* must have at least:
    ``chapter_id``, ``similarity``, and ``content`` (chunk text).
    Any extra keys (book_title, chapter_title, etc.) are preserved
    from the winning chunk.

    Returns:
        List of chapter-level result dicts sorted by descending similarity.
    """
    best: dict[str, dict] = {}

    for r in chunk_results:
        ch_id = r["chapter_id"]
        if ch_id not in best or r["similarity"] > best[ch_id]["similarity"]:
            entry = {k: v for k, v in r.items() if k not in ("chunk_id", "chunk_index")}
            entry["excerpt"] = r.get("content", "")
            best[ch_id] = entry

    results = list(best.values())
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
