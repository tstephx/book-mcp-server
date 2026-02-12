"""
Reading management tools for tracking progress and bookmarks
Enables personal library management with reading history and annotations
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ..database import get_db_connection, execute_query, execute_single

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_reading_tools(mcp: "FastMCP") -> None:
    """Register reading management tools with the MCP server"""

    @mcp.tool()
    def mark_as_read(
        book_id: str,
        chapter_number: int,
        notes: str = ""
    ) -> dict:
        """Mark a chapter as read

        Tracks reading progress for personal library management.
        Automatically records completion timestamp.

        Args:
            book_id: UUID of the book
            chapter_number: Chapter number that was read
            notes: Optional notes about the chapter

        Returns:
            Confirmation with updated reading stats

        Examples:
            mark_as_read("abc-123", 5)
            mark_as_read("abc-123", 5, notes="Great explanation of async patterns")
        """
        try:
            # Verify chapter exists
            chapter = execute_single("""
                SELECT c.id, c.title, b.title as book_title
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {"error": f"Chapter {chapter_number} not found in book {book_id}"}

            now = datetime.now().isoformat()

            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Upsert reading progress
                cursor.execute("""
                    INSERT INTO reading_progress (book_id, chapter_number, status, completed_at, notes)
                    VALUES (?, ?, 'read', ?, ?)
                    ON CONFLICT(book_id, chapter_number)
                    DO UPDATE SET status = 'read', completed_at = ?, notes = COALESCE(?, notes)
                """, (book_id, chapter_number, now, notes, now, notes if notes else None))

                conn.commit()

            # Get updated stats for this book
            stats = execute_single("""
                SELECT
                    COUNT(*) as total_chapters,
                    SUM(CASE WHEN rp.status = 'read' THEN 1 ELSE 0 END) as read_chapters
                FROM chapters c
                LEFT JOIN reading_progress rp ON c.book_id = rp.book_id AND c.chapter_number = rp.chapter_number
                WHERE c.book_id = ?
            """, (book_id,))

            logger.info(f"Marked as read: {chapter['book_title']} Ch.{chapter_number}")

            return {
                "success": True,
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number,
                "progress": {
                    "read": stats['read_chapters'] or 0,
                    "total": stats['total_chapters'],
                    "percent": round((stats['read_chapters'] or 0) / stats['total_chapters'] * 100, 1)
                }
            }

        except Exception as e:
            logger.error(f"mark_as_read error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def mark_as_reading(
        book_id: str,
        chapter_number: int
    ) -> dict:
        """Mark a chapter as currently being read

        Tracks what you're currently reading. Useful for resuming later.

        Args:
            book_id: UUID of the book
            chapter_number: Chapter number you're starting

        Returns:
            Confirmation with chapter details
        """
        try:
            chapter = execute_single("""
                SELECT c.id, c.title, b.title as book_title
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {"error": f"Chapter {chapter_number} not found in book {book_id}"}

            now = datetime.now().isoformat()

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO reading_progress (book_id, chapter_number, status, started_at)
                    VALUES (?, ?, 'reading', ?)
                    ON CONFLICT(book_id, chapter_number)
                    DO UPDATE SET status = 'reading', started_at = COALESCE(started_at, ?)
                """, (book_id, chapter_number, now, now))
                conn.commit()

            logger.info(f"Started reading: {chapter['book_title']} Ch.{chapter_number}")

            return {
                "success": True,
                "status": "reading",
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number
            }

        except Exception as e:
            logger.error(f"mark_as_reading error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def get_reading_progress(book_id: str = "") -> dict:
        """Get reading progress for a book or entire library

        Shows what you've read, what you're reading, and what's left.

        Args:
            book_id: Optional book UUID. If empty, returns progress for all books.

        Returns:
            Reading progress statistics and chapter status

        Examples:
            get_reading_progress()  # All books
            get_reading_progress("abc-123")  # Specific book
        """
        try:
            if book_id:
                # Get progress for specific book
                book = execute_single("SELECT id, title, author FROM books WHERE id = ?", (book_id,))
                if not book:
                    return {"error": f"Book not found: {book_id}"}

                chapters = execute_query("""
                    SELECT
                        c.chapter_number,
                        c.title,
                        c.word_count,
                        COALESCE(rp.status, 'unread') as status,
                        rp.completed_at,
                        rp.notes
                    FROM chapters c
                    LEFT JOIN reading_progress rp ON c.book_id = rp.book_id AND c.chapter_number = rp.chapter_number
                    WHERE c.book_id = ?
                    ORDER BY c.chapter_number
                """, (book_id,))

                read_count = sum(1 for c in chapters if c['status'] == 'read')
                reading_count = sum(1 for c in chapters if c['status'] == 'reading')
                total = len(chapters)

                return {
                    "book_title": book['title'],
                    "author": book['author'],
                    "progress": {
                        "read": read_count,
                        "reading": reading_count,
                        "unread": total - read_count - reading_count,
                        "total": total,
                        "percent_complete": round(read_count / total * 100, 1) if total > 0 else 0
                    },
                    "chapters": [
                        {
                            "number": c['chapter_number'],
                            "title": c['title'],
                            "status": c['status'],
                            "word_count": c['word_count'],
                            "completed_at": c['completed_at'],
                            "notes": c['notes']
                        }
                        for c in chapters
                    ]
                }

            else:
                # Get progress for all books
                books = execute_query("""
                    SELECT
                        b.id,
                        b.title,
                        b.author,
                        COUNT(c.id) as total_chapters,
                        SUM(CASE WHEN rp.status = 'read' THEN 1 ELSE 0 END) as read_chapters,
                        SUM(CASE WHEN rp.status = 'reading' THEN 1 ELSE 0 END) as reading_chapters
                    FROM books b
                    LEFT JOIN chapters c ON c.book_id = b.id
                    LEFT JOIN reading_progress rp ON c.book_id = rp.book_id AND c.chapter_number = rp.chapter_number
                    GROUP BY b.id
                    ORDER BY b.title
                """)

                total_read = sum(b['read_chapters'] or 0 for b in books)
                total_chapters = sum(b['total_chapters'] for b in books)

                # Currently reading
                currently_reading = execute_query("""
                    SELECT b.title as book_title, c.chapter_number, c.title as chapter_title
                    FROM reading_progress rp
                    JOIN books b ON rp.book_id = b.id
                    JOIN chapters c ON rp.book_id = c.book_id AND rp.chapter_number = c.chapter_number
                    WHERE rp.status = 'reading'
                    ORDER BY rp.started_at DESC
                """)

                return {
                    "library_progress": {
                        "total_chapters_read": total_read,
                        "total_chapters": total_chapters,
                        "percent_complete": round(total_read / total_chapters * 100, 1) if total_chapters > 0 else 0
                    },
                    "currently_reading": [
                        {
                            "book_title": r['book_title'],
                            "chapter_number": r['chapter_number'],
                            "chapter_title": r['chapter_title']
                        }
                        for r in currently_reading
                    ],
                    "books": [
                        {
                            "id": b['id'],
                            "title": b['title'],
                            "author": b['author'],
                            "read": b['read_chapters'] or 0,
                            "reading": b['reading_chapters'] or 0,
                            "total": b['total_chapters'],
                            "percent": round((b['read_chapters'] or 0) / b['total_chapters'] * 100, 1) if b['total_chapters'] > 0 else 0
                        }
                        for b in books
                    ]
                }

        except Exception as e:
            logger.error(f"get_reading_progress error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def add_bookmark(
        book_id: str,
        chapter_number: int,
        title: str = "",
        note: str = "",
        position: int = 0
    ) -> dict:
        """Add a bookmark to a chapter

        Save important passages for later reference.

        Args:
            book_id: UUID of the book
            chapter_number: Chapter to bookmark
            title: Short title for the bookmark
            note: Detailed note about why this is bookmarked
            position: Optional character position in chapter (default: 0 = start)

        Returns:
            Confirmation with bookmark details

        Examples:
            add_bookmark("abc-123", 5, title="Great async example")
            add_bookmark("abc-123", 5, note="Review this pattern for the project")
        """
        try:
            chapter = execute_single("""
                SELECT c.id, c.title, b.title as book_title
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {"error": f"Chapter {chapter_number} not found in book {book_id}"}

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO bookmarks (book_id, chapter_number, title, note, position)
                    VALUES (?, ?, ?, ?, ?)
                """, (book_id, chapter_number, title or chapter['title'], note, position))
                bookmark_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Added bookmark: {chapter['book_title']} Ch.{chapter_number}")

            return {
                "success": True,
                "bookmark_id": bookmark_id,
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number,
                "title": title or chapter['title'],
                "note": note
            }

        except Exception as e:
            logger.error(f"add_bookmark error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def get_bookmarks(book_id: str = "") -> dict:
        """Get all bookmarks, optionally filtered by book

        Retrieve saved bookmarks for quick reference.

        Args:
            book_id: Optional book UUID. If empty, returns all bookmarks.

        Returns:
            List of bookmarks with their details

        Examples:
            get_bookmarks()  # All bookmarks
            get_bookmarks("abc-123")  # Bookmarks for specific book
        """
        try:
            if book_id:
                book = execute_single("SELECT title FROM books WHERE id = ?", (book_id,))
                if not book:
                    return {"error": f"Book not found: {book_id}"}

                bookmarks = execute_query("""
                    SELECT
                        bm.id,
                        bm.chapter_number,
                        bm.title as bookmark_title,
                        bm.note,
                        bm.position,
                        bm.created_at,
                        c.title as chapter_title
                    FROM bookmarks bm
                    JOIN chapters c ON bm.book_id = c.book_id AND bm.chapter_number = c.chapter_number
                    WHERE bm.book_id = ?
                    ORDER BY bm.chapter_number, bm.created_at
                """, (book_id,))

                return {
                    "book_title": book['title'],
                    "count": len(bookmarks),
                    "bookmarks": [
                        {
                            "id": b['id'],
                            "chapter_number": b['chapter_number'],
                            "chapter_title": b['chapter_title'],
                            "title": b['bookmark_title'],
                            "note": b['note'],
                            "position": b['position'],
                            "created_at": b['created_at']
                        }
                        for b in bookmarks
                    ]
                }

            else:
                # All bookmarks grouped by book
                bookmarks = execute_query("""
                    SELECT
                        bm.id,
                        bm.book_id,
                        b.title as book_title,
                        bm.chapter_number,
                        bm.title as bookmark_title,
                        bm.note,
                        bm.position,
                        bm.created_at,
                        c.title as chapter_title
                    FROM bookmarks bm
                    JOIN books b ON bm.book_id = b.id
                    JOIN chapters c ON bm.book_id = c.book_id AND bm.chapter_number = c.chapter_number
                    ORDER BY bm.created_at DESC
                """)

                # Group by book
                by_book = {}
                for b in bookmarks:
                    book_id = b['book_id']
                    if book_id not in by_book:
                        by_book[book_id] = {
                            "book_title": b['book_title'],
                            "bookmarks": []
                        }
                    by_book[book_id]['bookmarks'].append({
                        "id": b['id'],
                        "chapter_number": b['chapter_number'],
                        "chapter_title": b['chapter_title'],
                        "title": b['bookmark_title'],
                        "note": b['note'],
                        "created_at": b['created_at']
                    })

                return {
                    "total_count": len(bookmarks),
                    "books_with_bookmarks": len(by_book),
                    "by_book": list(by_book.values()),
                    "recent": [
                        {
                            "book_title": b['book_title'],
                            "chapter_number": b['chapter_number'],
                            "title": b['bookmark_title'],
                            "created_at": b['created_at']
                        }
                        for b in bookmarks[:10]
                    ]
                }

        except Exception as e:
            logger.error(f"get_bookmarks error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def remove_bookmark(bookmark_id: int) -> dict:
        """Remove a bookmark

        Args:
            bookmark_id: ID of the bookmark to remove

        Returns:
            Confirmation of removal
        """
        try:
            bookmark = execute_single("""
                SELECT bm.id, b.title as book_title, bm.chapter_number, bm.title
                FROM bookmarks bm
                JOIN books b ON bm.book_id = b.id
                WHERE bm.id = ?
            """, (bookmark_id,))

            if not bookmark:
                return {"error": f"Bookmark not found: {bookmark_id}"}

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
                conn.commit()

            logger.info(f"Removed bookmark {bookmark_id}")

            return {
                "success": True,
                "removed": {
                    "id": bookmark_id,
                    "book_title": bookmark['book_title'],
                    "chapter_number": bookmark['chapter_number'],
                    "title": bookmark['title']
                }
            }

        except Exception as e:
            logger.error(f"remove_bookmark error: {e}", exc_info=True)
            return {"error": str(e)}
