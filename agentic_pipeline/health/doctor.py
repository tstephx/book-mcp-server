"""Integrity doctor — detect and repair library/pipeline drift.

Every check is a query that CAN return violations; the fixes reuse the
checks' SQL fragments so detector and repairer cannot drift apart.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

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


def _resolve_lost_source(source_path: str) -> str | None:
    """Original path, else PROCESSED_DIR/<basename>, else None.

    Basename matching is acceptable HERE (unlike resolve_source_file's
    hash-verified fallback for live books): a lost book has no live copy
    to corrupt — reingest mints a fresh record from whatever file exists.
    """
    if source_path and Path(source_path).exists():
        return source_path
    processed = os.environ.get("PROCESSED_DIR")
    if source_path and processed:
        candidate = Path(processed) / Path(source_path).name
        if candidate.exists():
            return str(candidate)
    return None


def check_lost_books(db_path) -> Finding:
    """Pipelines claiming COMPLETE while the library has no such book."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            """SELECT p.id, p.source_path FROM processing_pipelines p
               WHERE p.state = 'complete'
                 AND NOT EXISTS (SELECT 1 FROM books b WHERE b.id = p.id)"""
        ).fetchall()
        details = []
        for r in rows:
            source_path = r["source_path"] or ""
            basename = Path(source_path).name if source_path else ""
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE book_id = ?", (r["id"],)).fetchone()[0]
            sample_row = conn.execute(
                "SELECT content FROM chunks WHERE book_id = ? ORDER BY chunk_index LIMIT 1",
                (r["id"],),
            ).fetchone()
            sample = (sample_row["content"] or "")[:200] if sample_row else ""
            live_copy = bool(
                basename
                and conn.execute(
                    """SELECT 1 FROM processing_pipelines p2
                       JOIN books b ON b.id = p2.id
                       WHERE p2.source_path LIKE ? AND p2.id != ? LIMIT 1""",
                    (f"%{basename}", r["id"]),
                ).fetchone()
            )
            resolved = _resolve_lost_source(source_path)
            details.append(
                {
                    "pipeline_id": r["id"],
                    "source_path": source_path,
                    "basename": basename,
                    "chunk_count": chunk_count,
                    "source_available": resolved is not None,
                    "resolved_path": resolved,
                    "live_copy": live_copy,
                    "sample": sample,
                }
            )
    return Finding(
        category=CATEGORY_LOST_BOOKS,
        count=len(details),
        fixable_count=len(details),  # archiving fixes every lost book
        details=details,
    )
