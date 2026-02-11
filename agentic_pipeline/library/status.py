"""Library status dashboard - unified view of book library and pipeline state."""

import sqlite3
from pathlib import Path


class LibraryStatus:
    """Aggregates library status across books, chapters, and pipeline."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_status(self) -> dict:
        """Get unified library status."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            books = self._get_books(cursor)
        except sqlite3.OperationalError:
            # Tables don't exist yet (fresh init)
            conn.close()
            return self._empty_status()

        pipeline_summary = self._get_pipeline_summary(cursor)
        conn.close()

        # Compute per-book status
        book_list = []
        total_words = 0
        books_fully_ready = 0
        books_partially_ready = 0
        books_not_embedded = 0
        total_chapters = 0
        embedded_chapters = 0

        for row in books:
            chapters = row["chapter_count"] or 0
            embedded = row["embedded_count"] or 0
            total_chapters += chapters
            embedded_chapters += embedded
            total_words += row["word_count"] or 0

            if chapters > 0 and embedded == chapters:
                status = "ready"
                books_fully_ready += 1
            elif embedded > 0:
                status = "partial"
                books_partially_ready += 1
            else:
                status = "no_embeddings"
                books_not_embedded += 1

            embedding_pct = (embedded / chapters * 100) if chapters > 0 else 0.0

            book_list.append({
                "id": row["id"],
                "title": row["title"],
                "author": row["author"],
                "chapters": chapters,
                "embedded_chapters": embedded,
                "embedding_pct": round(embedding_pct, 1),
                "pipeline_state": row["pipeline_state"],
                "source": "pipeline" if row["pipeline_state"] else "direct",
                "status": status,
            })

        coverage_pct = (embedded_chapters / total_chapters * 100) if total_chapters > 0 else 0.0

        return {
            "overview": {
                "total_books": len(book_list),
                "total_chapters": total_chapters,
                "total_words": total_words,
                "books_fully_ready": books_fully_ready,
                "books_partially_ready": books_partially_ready,
                "books_not_embedded": books_not_embedded,
                "embedded_chapters": embedded_chapters,
                "embedding_coverage_pct": round(coverage_pct, 1),
            },
            "books": book_list,
            "pipeline_summary": pipeline_summary,
        }

    def _get_books(self, cursor: sqlite3.Cursor) -> list:
        """Query books joined with chapters and pipeline state."""
        cursor.execute("""
            SELECT b.id, b.title, b.author, b.word_count,
                   COUNT(c.id) as chapter_count,
                   SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_count,
                   pp.state as pipeline_state
            FROM books b
            LEFT JOIN chapters c ON c.book_id = b.id
            LEFT JOIN processing_pipelines pp ON pp.id = b.id
            GROUP BY b.id
            ORDER BY b.title
        """)
        return cursor.fetchall()

    def _get_pipeline_summary(self, cursor: sqlite3.Cursor) -> dict:
        """Get pipeline state counts."""
        try:
            cursor.execute("""
                SELECT state, COUNT(*) as cnt
                FROM processing_pipelines
                GROUP BY state
            """)
            rows = cursor.fetchall()
            by_state = {row["state"]: row["cnt"] for row in rows}
            total = sum(by_state.values())
            return {"total_pipelines": total, "by_state": by_state}
        except sqlite3.OperationalError:
            return {"total_pipelines": 0, "by_state": {}}

    def _empty_status(self) -> dict:
        """Return empty status when tables don't exist."""
        return {
            "overview": {
                "total_books": 0,
                "total_chapters": 0,
                "total_words": 0,
                "books_fully_ready": 0,
                "books_partially_ready": 0,
                "books_not_embedded": 0,
                "embedded_chapters": 0,
                "embedding_coverage_pct": 0.0,
            },
            "books": [],
            "pipeline_summary": {"total_pipelines": 0, "by_state": {}},
        }
