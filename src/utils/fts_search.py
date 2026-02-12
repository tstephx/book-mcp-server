"""
Full-Text Search using SQLite FTS5

Provides phrase search, boolean queries, and BM25 ranking.
Complements semantic search with exact term matching.
"""

import logging
import re
from typing import Optional

from ..database import get_db_connection, execute_query

logger = logging.getLogger(__name__)


def fts_table_exists() -> bool:
    """Check if FTS table is available"""
    try:
        rows = execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chapters_fts'"
        )
        return len(rows) > 0
    except Exception:
        return False


def escape_fts_query(query: str) -> str:
    """Escape special FTS5 characters in query

    FTS5 special chars: " * - ^ : ( )
    We preserve quoted phrases and basic operators.
    """
    # If query contains quotes, assume user knows what they're doing
    if '"' in query:
        return query

    # Escape special characters that could cause syntax errors
    # But preserve * for prefix matching
    special_chars = ['-', '^', ':', '(', ')']
    for char in special_chars:
        query = query.replace(char, f' ')

    # Clean up multiple spaces
    query = ' '.join(query.split())

    return query


def full_text_search(
    query: str,
    limit: int = 10,
    book_id: Optional[str] = None,
    highlight: bool = True
) -> dict:
    """Search chapter content using FTS5 full-text search

    Supports:
    - Phrase search: "async await"
    - Boolean: python AND async, python OR async
    - Prefix: python*
    - Negation: python NOT java

    Args:
        query: Search query (FTS5 syntax supported)
        limit: Maximum results (1-50, default: 10)
        book_id: Optional book ID to filter results
        highlight: Include highlighted excerpts (default: True)

    Returns:
        Dictionary with results and metadata
    """
    if not fts_table_exists():
        return {
            "error": "Full-text search not available. Run migrations/add_fts_and_summaries.py first.",
            "results": []
        }

    if not query or not query.strip():
        return {"error": "Query cannot be empty", "results": []}

    # Validate and clamp limit
    limit = max(1, min(50, limit))

    # Escape query for safety
    safe_query = escape_fts_query(query.strip())

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Build query with optional book filter
            if book_id:
                sql = """
                    SELECT
                        fts.chapter_id,
                        c.book_id,
                        c.chapter_number,
                        c.title,
                        b.title as book_title,
                        snippet(chapters_fts, 2, '**', '**', '...', 32) as excerpt,
                        bm25(chapters_fts) as rank
                    FROM chapters_fts fts
                    JOIN chapters c ON fts.chapter_id = c.id
                    JOIN books b ON c.book_id = b.id
                    WHERE chapters_fts MATCH ?
                      AND c.book_id = ?
                    ORDER BY rank
                    LIMIT ?
                """
                cursor.execute(sql, (safe_query, book_id, limit))
            else:
                sql = """
                    SELECT
                        fts.chapter_id,
                        c.book_id,
                        c.chapter_number,
                        c.title,
                        b.title as book_title,
                        snippet(chapters_fts, 2, '**', '**', '...', 32) as excerpt,
                        bm25(chapters_fts) as rank
                    FROM chapters_fts fts
                    JOIN chapters c ON fts.chapter_id = c.id
                    JOIN books b ON c.book_id = b.id
                    WHERE chapters_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                cursor.execute(sql, (safe_query, limit))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                result = {
                    'chapter_id': row['chapter_id'],
                    'book_id': row['book_id'],
                    'book_title': row['book_title'],
                    'chapter_number': row['chapter_number'],
                    'chapter_title': row['title'],
                    'rank': round(abs(row['rank']), 3)  # BM25 returns negative scores
                }

                if highlight:
                    result['excerpt'] = row['excerpt']

                results.append(result)

            logger.info(f"FTS search '{query}' returned {len(results)} results")

            return {
                "query": query,
                "results": results,
                "total_found": len(results)
            }

    except Exception as e:
        logger.error(f"FTS search error: {e}", exc_info=True)
        # Check if it's a syntax error
        if "fts5" in str(e).lower() or "syntax" in str(e).lower():
            return {
                "error": f"Invalid search syntax. Try simpler terms or use quotes for phrases.",
                "query": query,
                "results": []
            }
        return {"error": str(e), "results": []}


def rebuild_fts_index() -> dict:
    """Rebuild FTS index by repopulating from chapters table and files on disk.

    Deletes all existing FTS rows and re-indexes every chapter.
    Useful when the FTS index is stale (e.g. after DB migration or bulk import).
    """
    if not fts_table_exists():
        return {"error": "FTS table does not exist"}

    try:
        from .file_utils import read_chapter_content

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get count before
            cursor.execute("SELECT COUNT(*) as count FROM chapters_fts")
            before_count = cursor.fetchone()['count']

            # Clear existing FTS data
            cursor.execute("DELETE FROM chapters_fts")

            # Re-index all chapters from source
            cursor.execute("SELECT id, title, file_path FROM chapters")
            chapters = cursor.fetchall()

            indexed = 0
            errors = 0

            for chapter in chapters:
                try:
                    content = read_chapter_content(chapter['file_path'])
                    cursor.execute(
                        "INSERT INTO chapters_fts (chapter_id, title, content) VALUES (?, ?, ?)",
                        (chapter['id'], chapter['title'] or '', content)
                    )
                    indexed += 1

                    if indexed % 100 == 0:
                        conn.commit()
                        logger.info(f"FTS rebuild: indexed {indexed} chapters...")

                except Exception as e:
                    logger.warning(f"FTS rebuild: error indexing chapter {chapter['id']}: {e}")
                    errors += 1

            conn.commit()
            logger.info(f"FTS index rebuilt: {indexed} indexed, {errors} errors (was {before_count} entries)")

            return {
                "status": "rebuilt",
                "entries_before": before_count,
                "entries_after": indexed,
                "errors": errors
            }

    except Exception as e:
        logger.error(f"FTS rebuild error: {e}", exc_info=True)
        return {"error": str(e)}


def sync_fts_chapter(chapter_id: str, content: str, title: str = "") -> bool:
    """Sync a single chapter to FTS index

    Call this when chapter content is updated.

    Args:
        chapter_id: Chapter ID
        content: Full chapter content
        title: Chapter title

    Returns:
        True if successful
    """
    if not fts_table_exists():
        return False

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Delete existing entry
            cursor.execute(
                "DELETE FROM chapters_fts WHERE chapter_id = ?",
                (chapter_id,)
            )

            # Insert updated entry
            cursor.execute(
                "INSERT INTO chapters_fts (chapter_id, title, content) VALUES (?, ?, ?)",
                (chapter_id, title, content)
            )

            conn.commit()
            logger.debug(f"Synced chapter {chapter_id} to FTS index")
            return True

    except Exception as e:
        logger.error(f"FTS sync error for chapter {chapter_id}: {e}")
        return False
