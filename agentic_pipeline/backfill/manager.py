"""Backfill manager - registers legacy library books in the pipeline."""

import hashlib
import sqlite3
from pathlib import Path

from agentic_pipeline.db.pipelines import PipelineRepository


class BackfillManager:
    """Finds and backfills library books that have no pipeline record."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)

    def find_untracked(self) -> list[dict]:
        """Find library books without a pipeline record."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT b.id, b.title, b.author, b.source_file, b.word_count,
                       COUNT(c.id) as chapter_count,
                       SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_count
                FROM books b
                LEFT JOIN chapters c ON c.book_id = b.id
                LEFT JOIN processing_pipelines pp ON pp.id = b.id
                WHERE pp.id IS NULL
                GROUP BY b.id
                ORDER BY b.title
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def run(self, dry_run: bool = False) -> dict:
        """Backfill all untracked books.

        Args:
            dry_run: If True, report what would happen without changing anything.

        Returns:
            Dict with backfill results.
        """
        untracked = self.find_untracked()

        if dry_run:
            return {
                "backfilled": 0,
                "would_backfill": len(untracked),
                "skipped": 0,
                "books": [self._book_summary(b) for b in untracked],
            }

        backfilled = 0
        skipped = 0
        books = []

        for book in untracked:
            content_hash = self._compute_hash(book)
            created = self.repo.create_backfill(
                book_id=book["id"],
                source_path=book["source_file"] or "",
                content_hash=content_hash,
            )
            summary = self._book_summary(book)
            if created:
                backfilled += 1
                summary["action"] = "backfilled"
                self._write_audit(book["id"])
            else:
                skipped += 1
                summary["action"] = "skipped"
            books.append(summary)

        return {
            "backfilled": backfilled,
            "skipped": skipped,
            "books": books,
        }

    def _book_summary(self, book: dict) -> dict:
        """Create a summary dict for a book."""
        chapters = book["chapter_count"] or 0
        embedded = book["embedded_count"] or 0

        if chapters == 0:
            quality = "no_chapters"
        elif embedded < chapters:
            quality = "missing_embeddings"
        else:
            quality = "good"

        return {
            "id": book["id"],
            "title": book["title"],
            "author": book["author"],
            "chapter_count": chapters,
            "embedded_count": embedded,
            "word_count": book["word_count"] or 0,
            "quality": quality,
        }

    def _write_audit(self, book_id: str) -> None:
        """Record backfill in audit trail."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute(
                """
                INSERT INTO approval_audit
                (book_id, pipeline_id, action, actor, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (book_id, book_id, "backfill", "backfill:automated",
                 "Legacy book registered in pipeline"),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _compute_hash(book: dict) -> str:
        """Generate a content hash for a backfill book.

        Since we don't have the original file, hash the book_id + title.
        """
        data = f"{book['id']}:{book.get('title', '')}".encode()
        return f"backfill:{hashlib.sha256(data).hexdigest()}"
