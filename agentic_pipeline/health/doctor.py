"""Integrity doctor — detect and repair library/pipeline drift.

Every check is a query that CAN return violations; the fixes reuse the
checks' SQL fragments so detector and repairer cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_pipeline.db.connection import get_pipeline_db

logger = logging.getLogger(__name__)

CATEGORY_ORPHANED_CHUNKS = "orphaned_chunks"
CATEGORY_LOST_BOOKS = "lost_books"
CATEGORY_NULL_CONTENT_HASH = "null_content_hash"
CATEGORY_NULL_BOOK_TYPE = "null_book_type"

CATEGORIES = (
    CATEGORY_ORPHANED_CHUNKS,
    CATEGORY_LOST_BOOKS,
    CATEGORY_NULL_CONTENT_HASH,
    CATEGORY_NULL_BOOK_TYPE,
)

# Shared by check_orphaned_chunks and the delete fix — one definition of
# "orphan" so the two cannot disagree. Embedded or not is irrelevant:
# unjoinable rows are dead weight either way.
_ORPHAN_WHERE = """
    NOT EXISTS (SELECT 1 FROM chapters ch WHERE ch.id = chunks.chapter_id)
    OR NOT EXISTS (SELECT 1 FROM books b WHERE b.id = chunks.book_id)
"""


@dataclass
class Finding:
    """One category of integrity violation."""

    category: str
    count: int
    fixable_count: int
    details: list[dict] = field(default_factory=list)


def check_orphaned_chunks(db_path) -> Finding:
    """Chunks whose chapter_id or book_id resolves to nothing."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            f"""SELECT id AS chunk_id, chapter_id, book_id
                FROM chunks WHERE {_ORPHAN_WHERE}"""
        ).fetchall()
    details = [dict(r) for r in rows]
    return Finding(
        category=CATEGORY_ORPHANED_CHUNKS,
        count=len(details),
        fixable_count=len(details),  # deletion fixes every orphan
        details=details,
    )
