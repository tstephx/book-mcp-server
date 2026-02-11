"""
Processing Adapter - wraps book-ingestion library for the agentic pipeline.

This adapter provides a clean interface for the orchestrator to use
book-ingestion as a library instead of subprocess calls.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from book_ingestion import (
    BookIngestionApp,
    ProcessingMode,
    PipelineResult,
)
from book_ingestion.embeddings import EmbeddingGenerator

from agentic_pipeline.adapters.llm_fallback_adapter import LLMFallbackAdapter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from book processing through the adapter."""

    success: bool
    book_id: str
    quality_score: int
    detection_confidence: float
    detection_method: str
    needs_review: bool
    warnings: list[str]
    chapter_count: int
    word_count: int
    error: Optional[str] = None
    llm_fallback_used: bool = False


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    success: bool
    chapters_processed: int
    error: Optional[str] = None


class ProcessingAdapter:
    """
    Adapter that wraps BookIngestionApp for use by the agentic pipeline.

    Replaces subprocess calls with direct library imports for better
    performance, error handling, and data access.
    """

    def __init__(
        self,
        db_path: Path,
        output_dir: Optional[Path] = None,
        enable_llm_fallback: bool = True,
        llm_fallback_threshold: float = 0.5,
        processing_mode: str = "standard",
    ):
        """
        Initialize the processing adapter.

        Args:
            db_path: Path to the books SQLite database
            output_dir: Directory for processed book output
            enable_llm_fallback: Whether to enable LLM fallback for low confidence
            llm_fallback_threshold: Confidence below which LLM is triggered
            processing_mode: "quick", "standard", or "thorough"
        """
        self.db_path = Path(db_path)
        self.output_dir = output_dir or self.db_path.parent / "processed"

        # Create LLM fallback if enabled
        llm_fallback = None
        if enable_llm_fallback:
            llm_fallback = LLMFallbackAdapter(
                confidence_threshold=llm_fallback_threshold
            )

        # Create the book ingestion app
        self._app = BookIngestionApp.create(
            db_path=self.db_path,
            output_dir=self.output_dir,
            llm_fallback=llm_fallback,
            processing_mode=ProcessingMode(processing_mode),
            llm_fallback_threshold=llm_fallback_threshold,
        )

        # Lazy-loaded embedding generator
        self._embedding_generator: Optional[EmbeddingGenerator] = None

    def process_book(
        self,
        book_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        book_id: Optional[str] = None,
        save_to_storage: bool = True,
    ) -> ProcessingResult:
        """
        Process a book file and return structured results.

        Args:
            book_path: Path to the book file (PDF or EPUB)
            title: Optional title override
            author: Optional author override
            book_id: Optional book ID (generated if not provided)

        Returns:
            ProcessingResult with processing details
        """
        logger.info(f"Processing book: {book_path}")

        try:
            result = self._app.process(
                file_path=book_path,
                title=title,
                author=author,
                book_id=book_id,
                save_to_storage=save_to_storage,
            )

            if not result.success:
                return ProcessingResult(
                    success=False,
                    book_id=result.book_id,
                    quality_score=0,
                    detection_confidence=0.0,
                    detection_method="",
                    needs_review=True,
                    warnings=[],
                    chapter_count=0,
                    word_count=0,
                    error=result.error,
                )

            # Extract data from pipeline result
            pipeline_result = result.pipeline_result
            return ProcessingResult(
                success=True,
                book_id=result.book_id,
                quality_score=pipeline_result.quality_report.quality_score,
                detection_confidence=pipeline_result.detection_confidence,
                detection_method=pipeline_result.detection_method,
                needs_review=pipeline_result.needs_review,
                warnings=pipeline_result.warnings,
                chapter_count=len(pipeline_result.chapters),
                word_count=sum(
                    ch.get("word_count", 0) for ch in pipeline_result.chapters
                ),
                llm_fallback_used=result.llm_fallback_used,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                book_id=book_id or "",
                quality_score=0,
                detection_confidence=0.0,
                detection_method="",
                needs_review=True,
                warnings=[],
                chapter_count=0,
                word_count=0,
                error=str(e),
            )

    def generate_embeddings(
        self,
        book_id: Optional[str] = None,
        batch_size: int = 32,
    ) -> EmbeddingResult:
        """
        Generate embeddings for book chapters using direct SQL.

        If book_id is provided, generates embeddings only for that book.
        Otherwise, generates embeddings for all chapters without embeddings.

        Args:
            book_id: Optional book ID to limit embedding generation
            batch_size: Number of chapters to process at once

        Returns:
            EmbeddingResult with generation details
        """
        import hashlib
        import io
        import sqlite3
        from datetime import datetime, timezone

        import numpy as np

        try:
            # Lazy-load embedding generator
            if self._embedding_generator is None:
                self._embedding_generator = EmbeddingGenerator()

            conn = sqlite3.connect(str(self.db_path), timeout=10)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if book_id:
                    cursor.execute(
                        "SELECT id, file_path FROM chapters "
                        "WHERE book_id = ? AND embedding IS NULL",
                        (book_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT id, file_path FROM chapters WHERE embedding IS NULL"
                    )

                chapters = cursor.fetchall()

                if not chapters:
                    return EmbeddingResult(success=True, chapters_processed=0)

                logger.info(f"Generating embeddings for {len(chapters)} chapters")
                books_dir = self.db_path.parent / "books"
                processed = 0

                for i in range(0, len(chapters), batch_size):
                    batch = chapters[i : i + batch_size]
                    texts = []
                    valid = []

                    for ch in batch:
                        try:
                            content = self._read_chapter_content(
                                ch["file_path"], books_dir
                            )
                            if content.strip():
                                texts.append(content)
                                valid.append(ch)
                        except Exception as e:
                            logger.warning(f"Cannot read chapter {ch['id']}: {e}")

                    if not texts:
                        continue

                    embeddings = self._embedding_generator.generate_batch(
                        texts, batch_size=batch_size
                    )

                    now = datetime.now(timezone.utc).isoformat()

                    for ch, content, emb in zip(valid, texts, embeddings):
                        emb_blob = io.BytesIO()
                        np.save(emb_blob, emb)
                        content_hash = hashlib.sha256(content.encode()).hexdigest()
                        file_mtime = self._get_file_mtime(
                            ch["file_path"], books_dir
                        )

                        cursor.execute(
                            """
                            UPDATE chapters
                            SET embedding = ?, embedding_model = ?, content_hash = ?,
                                file_mtime = ?, embedding_updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                emb_blob.getvalue(),
                                "all-MiniLM-L6-v2",
                                content_hash,
                                file_mtime,
                                now,
                                ch["id"],
                            ),
                        )
                        processed += 1

                    conn.commit()

                return EmbeddingResult(success=True, chapters_processed=processed)
            finally:
                conn.close()

        except ImportError as e:
            logger.warning(f"Embedding dependencies not available: {e}")
            return EmbeddingResult(
                success=False,
                chapters_processed=0,
                error=f"Embedding dependencies not available: {e}",
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return EmbeddingResult(
                success=False,
                chapters_processed=0,
                error=str(e),
            )

    @staticmethod
    def _read_chapter_content(file_path: str, books_dir: Path) -> str:
        """Read chapter content from file, handling split chapters."""
        path = Path(file_path)

        if not path.is_absolute():
            # Resolve relative paths (e.g. data/books/...) against books_dir
            try:
                rel = path.relative_to("data/books")
                path = books_dir / rel
            except ValueError:
                path = books_dir / path

        if path.is_file():
            return path.read_text(encoding="utf-8")

        # Handle split chapters (directory with numbered .md parts)
        dir_path = path if path.is_dir() else path.with_suffix("")
        if dir_path.is_dir():
            parts = sorted(
                p for p in dir_path.glob("[0-9]*.md")
                if not p.name.startswith("_")
            )
            if not parts:
                parts = sorted(
                    p for p in dir_path.glob("*.md")
                    if not p.name.startswith("_")
                )
            if parts:
                return "\n\n".join(p.read_text(encoding="utf-8") for p in parts)

        raise FileNotFoundError(f"Chapter not found: {file_path}")

    @staticmethod
    def _get_file_mtime(file_path: str, books_dir: Path) -> float:
        """Get chapter file modification time. Returns 0 on any error."""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                try:
                    rel = path.relative_to("data/books")
                    path = books_dir / rel
                except ValueError:
                    path = books_dir / path

            if path.is_dir():
                mtimes = [p.stat().st_mtime for p in path.glob("*.md")]
                return max(mtimes) if mtimes else 0
            elif path.is_file():
                return path.stat().st_mtime
            return 0
        except Exception:
            return 0

    def get_processing_status(self, book_id: str) -> Optional[dict]:
        """
        Get the processing status for a book.

        Args:
            book_id: The book ID to check

        Returns:
            Dictionary with book status, or None if not found
        """
        from book_ingestion import BookDatabase

        db = BookDatabase(str(self.db_path))
        book = db.get_book(book_id)

        if not book:
            return None

        chapters = db.get_chapters_by_book(book_id)

        return {
            "book_id": book_id,
            "title": book.get("title"),
            "author": book.get("author"),
            "status": book.get("processing_status"),
            "chapter_count": len(chapters),
            "word_count": book.get("word_count", 0),
            "has_embeddings": any(
                ch.get("has_embedding", False) for ch in chapters
            ),
        }
