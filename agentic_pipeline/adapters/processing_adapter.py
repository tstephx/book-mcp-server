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
                save_to_storage=True,
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
        Generate embeddings for book chapters.

        If book_id is provided, generates embeddings only for that book.
        Otherwise, generates embeddings for all chapters without embeddings.

        Args:
            book_id: Optional book ID to limit embedding generation
            batch_size: Number of chapters to process at once

        Returns:
            EmbeddingResult with generation details
        """
        try:
            # Lazy-load embedding generator
            if self._embedding_generator is None:
                self._embedding_generator = EmbeddingGenerator()

            # Get chapters needing embeddings from database
            from book_ingestion import BookDatabase

            db = BookDatabase(str(self.db_path))

            if book_id:
                chapters = db.get_chapters_by_book(book_id)
            else:
                # Get all chapters without embeddings
                chapters = db.get_chapters_without_embeddings()

            if not chapters:
                return EmbeddingResult(
                    success=True,
                    chapters_processed=0,
                )

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chapters)} chapters")
            results = self._embedding_generator.generate_for_chapters(
                chapters,
                text_key="content",
                id_key="id",
                batch_size=batch_size,
            )

            # Store embeddings in database
            for embedding_result in results:
                db.store_embedding(
                    chapter_id=embedding_result.chapter_id,
                    embedding=embedding_result.embedding,
                    model_name=embedding_result.model_name,
                )

            return EmbeddingResult(
                success=True,
                chapters_processed=len(results),
            )

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
