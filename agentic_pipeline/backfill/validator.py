"""Library validator - checks books for quality issues."""

from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db


class LibraryValidator:
    """Validates library books for common quality issues."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)

    def validate(self) -> list[dict]:
        """Check all books for quality issues.

        Returns a list of issue dicts with book_id, title, and issue type.
        """
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT b.id, b.title, b.word_count,
                       COUNT(c.id) as chapter_count,
                       SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_count
                FROM books b
                LEFT JOIN chapters c ON c.book_id = b.id
                GROUP BY b.id
            """)

            issues = []
            for row in cursor.fetchall():
                book = dict(row)
                chapters = book["chapter_count"] or 0
                embedded = book["embedded_count"] or 0

                if chapters == 0:
                    issues.append({
                        "book_id": book["id"],
                        "title": book["title"],
                        "issue": "no_chapters",
                        "detail": "Book has no chapters",
                    })
                elif embedded < chapters:
                    issues.append({
                        "book_id": book["id"],
                        "title": book["title"],
                        "issue": "missing_embeddings",
                        "detail": f"{embedded}/{chapters} chapters embedded",
                    })

                if chapters > 0 and (book["word_count"] or 0) < 1000:
                    issues.append({
                        "book_id": book["id"],
                        "title": book["title"],
                        "issue": "low_word_count",
                        "detail": f"{book['word_count'] or 0} words",
                    })

            return issues
