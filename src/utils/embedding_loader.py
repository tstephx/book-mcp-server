"""Shared embedding loading utility

Loads chapter embeddings from DB or cache. Used by both semantic search
and hybrid search tools to avoid duplicating the loading logic.
"""

import io
import logging
from typing import Optional

import numpy as np

from ..database import get_db_connection
from .cache import get_cache

logger = logging.getLogger(__name__)


def load_chapter_embeddings(
    cache=None,
) -> tuple[Optional[np.ndarray], Optional[list[dict]]]:
    """Load chapter embeddings from cache or database.

    Args:
        cache: LibraryCache instance (uses global cache if None)

    Returns:
        Tuple of (embeddings_matrix, chapter_metadata) or (None, None)
        if no embeddings found. Metadata dicts have keys:
        id, book_id, book_title, chapter_title, chapter_number, file_path
    """
    if cache is None:
        cache = get_cache()

    cached = cache.get_embeddings()
    if cached:
        return cached

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                c.id,
                c.book_id,
                c.chapter_number,
                c.title as chapter_title,
                c.embedding,
                c.file_path,
                b.title as book_title
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.id
        """)
        rows = cursor.fetchall()

    if not rows:
        logger.warning("No embeddings found in database")
        return None, None

    chapter_embeddings = []
    chapter_metadata = []

    for row in rows:
        embedding = np.load(io.BytesIO(row['embedding']))
        chapter_embeddings.append(embedding)
        chapter_metadata.append({
            'id': row['id'],
            'book_id': row['book_id'],
            'book_title': row['book_title'],
            'chapter_title': row['chapter_title'],
            'chapter_number': row['chapter_number'],
            'file_path': row['file_path'],
        })

    embeddings_matrix = np.vstack(chapter_embeddings)
    cache.set_embeddings(embeddings_matrix, chapter_metadata)

    logger.info(f"Loaded {len(chapter_metadata)} chapter embeddings from DB")
    return embeddings_matrix, chapter_metadata
