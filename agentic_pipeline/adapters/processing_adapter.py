"""
Processing Adapter - wraps book-ingestion library for the agentic pipeline.

This adapter provides a clean interface for the orchestrator to use
book-ingestion as a library instead of subprocess calls.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from book_ingestion import BookIngestionApp, ProcessingMode
from src.utils.chunker import chunk_chapter
from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

from agentic_pipeline.adapters.llm_fallback_adapter import LLMFallbackAdapter
from agentic_pipeline.db.connection import get_pipeline_db
from agentic_pipeline.library.chapter_reader import read_chapter_content

logger = logging.getLogger(__name__)

BOOK_PROFILE_COLUMNS = (
    ("book_type", "TEXT"),
    ("classification_confidence", "REAL"),
    ("suggested_tags", "TEXT"),
    ("classification_reasoning", "TEXT"),
    ("classified_at", "TEXT"),
    ("classified_by", "TEXT"),
)


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
    chunks_created: int = 0
    chunks_embedded: int = 0
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

        # Store config for per-call app creation (BookIngestionApp holds a SQLite
        # connection; creating it here would bind it to the main thread and fail
        # when process_book() is called from a ThreadPoolExecutor worker thread)
        self._enable_llm_fallback = enable_llm_fallback
        self._llm_fallback_threshold = llm_fallback_threshold
        self._processing_mode = processing_mode

        # Lazy-loaded embedding generator
        self._embedding_generator: Optional[OpenAIEmbeddingGenerator] = None

    def process_book(
        self,
        book_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        book_id: Optional[str] = None,
        save_to_storage: bool = True,
        force_fallback: bool = False,
        book_profile: Optional[dict] = None,
    ) -> ProcessingResult:
        """
        Process a book file and return structured results.

        Args:
            book_path: Path to the book file (PDF or EPUB)
            title: Optional title override
            author: Optional author override
            book_id: Optional book ID (generated if not provided)
            force_fallback: If True, force LLM fallback processing
            book_profile: Optional classifier output to copy onto the stored book row

        Returns:
            ProcessingResult with processing details
        """
        logger.info(f"Processing book: {book_path}")

        try:
            # Create app in the calling thread — BookIngestionApp holds a SQLite
            # connection and cannot be shared across threads
            llm_fallback = None
            if self._enable_llm_fallback:
                llm_fallback = LLMFallbackAdapter(confidence_threshold=self._llm_fallback_threshold)
            app = BookIngestionApp.create(
                db_path=self.db_path,
                output_dir=self.output_dir,
                llm_fallback=llm_fallback,
                processing_mode=ProcessingMode(self._processing_mode),
                llm_fallback_threshold=self._llm_fallback_threshold,
            )
            result = app.process(
                file_path=book_path,
                title=title,
                author=author,
                book_id=book_id,
                save_to_storage=save_to_storage,
                force_fallback=force_fallback,
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

            if save_to_storage and book_profile is not None:
                self._persist_book_profile(result.book_id, book_profile)

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
                word_count=sum(ch.get("word_count", 0) for ch in pipeline_result.chapters),
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

    def _persist_book_profile(self, book_id: str, book_profile: dict) -> None:
        """Copy pipeline classification data onto the newly stored library row."""
        with get_pipeline_db(str(self.db_path)) as conn:
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(books)").fetchall()
            }
            for column_name, column_type in BOOK_PROFILE_COLUMNS:
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE books ADD COLUMN {column_name} {column_type}"
                    )

            cursor = conn.execute(
                """UPDATE books SET
                    book_type = ?,
                    classification_confidence = ?,
                    suggested_tags = ?,
                    classification_reasoning = ?,
                    classified_at = ?,
                    classified_by = ?
                WHERE id = ?""",
                (
                    book_profile.get("book_type", "unknown"),
                    book_profile.get("confidence", 0.0),
                    json.dumps(book_profile.get("suggested_tags", [])),
                    book_profile.get("reasoning", ""),
                    datetime.now(timezone.utc).isoformat(),
                    "agentic_pipeline",
                    book_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Stored book row not found for classification: {book_id}")
            conn.commit()

    def generate_embeddings(
        self,
        book_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> EmbeddingResult:
        """
        Chunk chapters and generate OpenAI embeddings for each chunk.

        If book_id is provided, processes only that book's chapters.
        Otherwise, processes all chapters that don't yet have chunks.

        Args:
            book_id: Optional book ID to limit processing
            batch_size: Number of chunks per OpenAI API call

        Returns:
            EmbeddingResult with generation details
        """
        import hashlib
        import io

        import numpy as np

        try:
            # Lazy-load embedding generator
            if self._embedding_generator is None:
                self._embedding_generator = OpenAIEmbeddingGenerator()

            with get_pipeline_db(str(self.db_path)) as conn:
                cursor = conn.cursor()

                # Find chapters that need chunking (no chunks exist yet)
                if book_id:
                    cursor.execute(
                        """SELECT c.id, c.book_id, c.file_path FROM chapters c
                        WHERE c.book_id = ?
                        AND NOT EXISTS (
                            SELECT 1 FROM chunks k WHERE k.chapter_id = c.id
                        )""",
                        (book_id,),
                    )
                else:
                    cursor.execute(
                        """SELECT c.id, c.book_id, c.file_path FROM chapters c
                        WHERE NOT EXISTS (
                            SELECT 1 FROM chunks k WHERE k.chapter_id = c.id
                        )"""
                    )

                chapters = cursor.fetchall()

                if not chapters:
                    return EmbeddingResult(success=True, chapters_processed=0)

                logger.info(f"Chunking and embedding {len(chapters)} chapters")
                books_dir = self.db_path.parent / "books"
                chapters_processed = 0

                # Phase 1: Chunk all chapters and insert rows
                all_chunk_rows = []

                for ch in chapters:
                    try:
                        content = read_chapter_content(ch["file_path"], books_dir)
                        if not content.strip():
                            continue

                        chunks = chunk_chapter(content)
                        for chunk in chunks:
                            chunk_id = f"{ch['id']}:{chunk['chunk_index']}"
                            content_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
                            all_chunk_rows.append(
                                (
                                    chunk_id,
                                    ch["id"],
                                    ch["book_id"],
                                    chunk["chunk_index"],
                                    chunk["content"],
                                    chunk["word_count"],
                                    content_hash,
                                )
                            )

                        chapters_processed += 1

                    except Exception as e:
                        logger.warning(f"Cannot chunk chapter {ch['id']}: {e}")

                if not all_chunk_rows:
                    return EmbeddingResult(success=True, chapters_processed=chapters_processed)

                # Insert chunk rows
                cursor.executemany(
                    """INSERT OR REPLACE INTO chunks
                    (id, chapter_id, book_id, chunk_index, content, word_count, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    all_chunk_rows,
                )
                total_chunks_created = len(all_chunk_rows)
                conn.commit()

                # Phase 2: Embed chunks in batches
                chunk_texts = [row[4] for row in all_chunk_rows]
                chunk_ids = [row[0] for row in all_chunk_rows]
                chunks_embedded = 0

                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i : i + batch_size]
                    batch_ids = chunk_ids[i : i + batch_size]

                    embeddings = self._embedding_generator.generate_batch(batch_texts)

                    for cid, emb in zip(batch_ids, embeddings):
                        buf = io.BytesIO()
                        np.save(buf, emb)
                        cursor.execute(
                            """UPDATE chunks SET embedding = ?, embedding_model = ?
                            WHERE id = ?""",
                            (buf.getvalue(), "text-embedding-3-large", cid),
                        )
                        chunks_embedded += 1

                    conn.commit()

                logger.info(f"Embedded {chunks_embedded} chunks from {chapters_processed} chapters")
                return EmbeddingResult(
                    success=True,
                    chapters_processed=chapters_processed,
                    chunks_created=total_chunks_created,
                    chunks_embedded=chunks_embedded,
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
            "has_embeddings": any(ch.get("has_embedding", False) for ch in chapters),
        }
