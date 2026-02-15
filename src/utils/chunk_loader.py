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
