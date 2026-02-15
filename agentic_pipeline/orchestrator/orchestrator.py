# agentic_pipeline/orchestrator/orchestrator.py
"""Pipeline Orchestrator - coordinates book processing."""

import hashlib
import logging
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from agentic_pipeline.config import OrchestratorConfig
from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState, TERMINAL_STATES
from agentic_pipeline.orchestrator.logging import PipelineLogger
from agentic_pipeline.orchestrator.errors import (
    IdempotencyError,
    ProcessingError,
    EmbeddingError,
    PipelineTimeoutError,
)
from agentic_pipeline.agents.classifier import ClassifierAgent
from agentic_pipeline.pipeline.strategy import StrategySelector
from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter
from agentic_pipeline.validation import ExtractionValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates book processing through the pipeline."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.repo = PipelineRepository(config.db_path)
        self.logger = PipelineLogger()
        self.classifier = ClassifierAgent(config.db_path)
        self.strategy_selector = StrategySelector()
        self.shutdown_requested = False
        self._seen_paths: set[str] = set()

        # Initialize processing adapter for direct library calls
        self.processing_adapter = ProcessingAdapter(
            db_path=config.db_path,
            enable_llm_fallback=True,
            llm_fallback_threshold=config.confidence_threshold,
        )

    def _compute_hash(self, book_path: str) -> str:
        """Compute SHA-256 hash of file contents."""
        path = Path(book_path)
        if not path.exists():
            raise FileNotFoundError(f"Book not found: {book_path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _check_idempotency(self, content_hash: str) -> Optional[dict]:
        """Check if book already exists. Returns existing record or None."""
        existing = self.repo.find_by_hash(content_hash)
        if not existing:
            return None

        state = PipelineState(existing["state"])

        if state == PipelineState.COMPLETE:
            return {"skipped": True, "reason": "Already processed", "pipeline_id": existing["id"]}

        if state not in TERMINAL_STATES:
            return {"skipped": True, "reason": "Already in progress", "pipeline_id": existing["id"]}

        # Terminal but not complete (rejected, archived) - allow reprocessing
        return None

    def _extract_sample(self, book_path: str, max_chars: int = 40000) -> str:
        """Extract text sample from book for classification."""
        path = Path(book_path)
        suffix = path.suffix.lower()

        # Handle epub files
        if suffix == ".epub":
            return self._extract_epub_text(path, max_chars)

        # Handle PDF files (basic - would need PyMuPDF for full support)
        if suffix == ".pdf":
            return f"[PDF file: {path.name} - requires processing pipeline]"

        # Try reading as text
        try:
            return path.read_text(encoding="utf-8")[:max_chars]
        except UnicodeDecodeError:
            return f"[Binary file: {path.name}]"

    def _extract_epub_text(self, path: Path, max_chars: int = 40000) -> str:
        """Extract text from epub file."""
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class HTMLTextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                def handle_data(self, data):
                    self.text.append(data)
                def get_text(self):
                    return " ".join(self.text)

            book = epub.read_epub(str(path), options={"ignore_ncx": True})
            text_parts = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    parser = HTMLTextExtractor()
                    content = item.get_content().decode("utf-8", errors="ignore")
                    parser.feed(content)
                    text_parts.append(parser.get_text())

                    # Stop if we have enough text
                    if sum(len(p) for p in text_parts) > max_chars:
                        break

            result = "\n\n".join(text_parts)[:max_chars]
            return result if result.strip() else f"[Empty epub: {path.name}]"

        except Exception as e:
            return f"[Epub extraction failed: {path.name} - {e}]"

    def _transition(self, pipeline_id: str, to_state: PipelineState):
        """Transition pipeline to new state with logging."""
        current = self.repo.get(pipeline_id)
        from_state = current["state"] if current else "none"
        self.repo.update_state(pipeline_id, to_state)
        self.logger.state_transition(pipeline_id, from_state, to_state.value)

    def _run_classifier(self, pipeline_id: str, text: str, content_hash: str) -> dict:
        """Run classification and store result."""
        profile = self.classifier.classify(text, content_hash)
        profile_dict = profile.to_dict()
        self.repo.update_book_profile(pipeline_id, profile_dict)
        return profile_dict

    def _run_processing(self, book_path: str, book_id: Optional[str] = None, force_fallback: bool = False) -> dict:
        """Run book-ingestion processing via direct library call.

        Args:
            book_path: Path to the book file.
            book_id: Optional pipeline ID to use as book ID.
            force_fallback: If True, force LLM fallback processing.

        Returns:
            Dict with processing results including quality_score, confidence, etc.
        """
        result = self.processing_adapter.process_book(
            book_path=book_path,
            book_id=book_id,
            force_fallback=force_fallback,
        )

        if not result.success:
            raise ProcessingError(
                f"Processing failed: {result.error}",
                exit_code=1,
                stderr=result.error or "Unknown error",
            )

        return {
            "book_id": result.book_id,
            "quality_score": result.quality_score,
            "detection_confidence": result.detection_confidence,
            "detection_method": result.detection_method,
            "needs_review": result.needs_review,
            "warnings": result.warnings,
            "chapter_count": result.chapter_count,
            "word_count": result.word_count,
            "llm_fallback_used": result.llm_fallback_used,
        }

    def _run_embedding(self, book_id: Optional[str] = None) -> dict:
        """Run embedding generation via direct library call.

        Args:
            book_id: Optional book ID to limit embedding to specific book

        Returns:
            Dict with embedding results
        """
        result = self.processing_adapter.generate_embeddings(book_id=book_id)

        if not result.success:
            raise EmbeddingError(
                f"Embedding failed: {result.error}",
                exit_code=1,
            )

        return {
            "chapters_processed": result.chapters_processed,
        }

    def process_one(self, book_path: str) -> dict:
        """Process a single book through the pipeline."""
        start_time = time.time()
        self.logger.processing_started("pending", book_path)

        # Compute hash and check idempotency
        content_hash = self._compute_hash(book_path)
        idempotency_check = self._check_idempotency(content_hash)
        if idempotency_check:
            return idempotency_check

        # Create pipeline record
        pipeline_id = self.repo.create(book_path, content_hash)

        try:
            # Process through states
            result = self._process_book(pipeline_id, book_path, content_hash)
            duration = time.time() - start_time
            self.logger.processing_complete(pipeline_id, duration)
            return result

        except Exception as e:
            self.logger.error(pipeline_id, type(e).__name__, str(e))
            self.repo.update_state(pipeline_id, PipelineState.NEEDS_RETRY)
            return {
                "pipeline_id": pipeline_id,
                "state": PipelineState.NEEDS_RETRY.value,
                "error": str(e)
            }

    def _process_book(self, pipeline_id: str, book_path: str, content_hash: str) -> dict:
        """Process book through all states."""

        # HASHING (already done via _compute_hash)
        self._transition(pipeline_id, PipelineState.HASHING)

        # Check duplicate
        # (For now, skip - hash uniqueness is enforced by DB)

        # CLASSIFYING
        self._transition(pipeline_id, PipelineState.CLASSIFYING)
        text_sample = self._extract_sample(book_path)
        profile = self._run_classifier(pipeline_id, text_sample, content_hash)

        # SELECTING_STRATEGY
        self._transition(pipeline_id, PipelineState.SELECTING_STRATEGY)
        strategy = self.strategy_selector.select(profile)
        self.repo.update_strategy_config(pipeline_id, strategy)

        # PROCESSING
        self._transition(pipeline_id, PipelineState.PROCESSING)
        try:
            processing_result = self._run_processing(book_path, book_id=pipeline_id)
        except (ProcessingError, PipelineTimeoutError) as e:
            self.logger.error(pipeline_id, type(e).__name__, str(e))
            self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
            return {"pipeline_id": pipeline_id, "state": PipelineState.NEEDS_RETRY.value, "error": str(e)}

        # Store processing results in pipeline record
        self.repo.update_processing_result(pipeline_id, {
            "quality_score": processing_result.get("quality_score"),
            "detection_confidence": processing_result.get("detection_confidence"),
            "detection_method": processing_result.get("detection_method"),
            "chapter_count": processing_result.get("chapter_count"),
            "word_count": processing_result.get("word_count"),
            "warnings": processing_result.get("warnings", []),
            "llm_fallback_used": processing_result.get("llm_fallback_used", False),
        })

        # VALIDATING
        self._transition(pipeline_id, PipelineState.VALIDATING)
        validator = ExtractionValidator()
        validation = validator.validate(
            book_id=pipeline_id, db_path=str(self.config.db_path)
        )

        if not validation.passed:
            # Check if we can retry with force_fallback
            current = self.repo.get(pipeline_id)
            retry_count = current.get("retry_count", 0) if current else 0
            first_attempt = {"reasons": validation.reasons, "metrics": validation.metrics}

            if retry_count == 0:
                # First failure — retry with force_fallback
                reason = "; ".join(validation.reasons)
                self.logger.error(pipeline_id, "ValidationFailed", f"{reason}. Retrying with force_fallback")
                self.repo.increment_retry_count(pipeline_id)

                # Transition through NEEDS_RETRY back to PROCESSING
                self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
                self._transition(pipeline_id, PipelineState.PROCESSING)

                try:
                    processing_result = self._run_processing(book_path, book_id=pipeline_id, force_fallback=True)
                except (ProcessingError, PipelineTimeoutError) as e:
                    # Retry processing failed — reject directly (max 1 retry)
                    self.logger.error(pipeline_id, type(e).__name__, str(e))
                    self.logger.state_transition(pipeline_id, PipelineState.PROCESSING.value, PipelineState.REJECTED.value)
                    self.repo.update_state(
                        pipeline_id,
                        PipelineState.REJECTED,
                        error_details={
                            "first_attempt": first_attempt,
                            "retry_error": str(e),
                        },
                    )
                    return {"pipeline_id": pipeline_id, "state": PipelineState.REJECTED.value, "error": str(e)}

                # Store updated processing results
                self.repo.update_processing_result(pipeline_id, {
                    "quality_score": processing_result.get("quality_score"),
                    "detection_confidence": processing_result.get("detection_confidence"),
                    "detection_method": processing_result.get("detection_method"),
                    "chapter_count": processing_result.get("chapter_count"),
                    "word_count": processing_result.get("word_count"),
                    "warnings": processing_result.get("warnings", []),
                    "llm_fallback_used": processing_result.get("llm_fallback_used", False),
                })

                # Re-validate
                self._transition(pipeline_id, PipelineState.VALIDATING)
                validation = validator.validate(
                    book_id=pipeline_id, db_path=str(self.config.db_path)
                )

            if not validation.passed:
                reason = "; ".join(validation.reasons)
                self.logger.error(pipeline_id, "ValidationFailed", reason)
                self.logger.state_transition(pipeline_id, PipelineState.VALIDATING.value, PipelineState.REJECTED.value)
                self.repo.update_state(
                    pipeline_id,
                    PipelineState.REJECTED,
                    error_details={
                        "validation_reasons": validation.reasons,
                        "metrics": validation.metrics,
                        "first_attempt": first_attempt,
                    },
                )
                return {
                    "pipeline_id": pipeline_id,
                    "state": PipelineState.REJECTED.value,
                    "reason": reason,
                    "metrics": validation.metrics,
                }

        # APPROVAL ROUTING - use processing result confidence
        confidence = processing_result.get("detection_confidence", 0)
        needs_review = processing_result.get("needs_review", False)

        # Route through PENDING_APPROVAL for both paths
        self._transition(pipeline_id, PipelineState.PENDING_APPROVAL)

        if confidence >= self.config.confidence_threshold and not needs_review:
            # Auto-approve
            self._transition(pipeline_id, PipelineState.APPROVED)
            self.repo.mark_approved(pipeline_id, approved_by="auto:high_confidence", confidence=confidence)
        else:
            # Needs human review — notify via macOS notification
            book_name = Path(book_path).stem
            self._notify_pending_approval(book_name, confidence)
            return {
                "pipeline_id": pipeline_id,
                "state": PipelineState.PENDING_APPROVAL.value,
                "book_type": profile.get("book_type"),
                "confidence": confidence,
                "needs_review": True
            }

        # EMBEDDING → COMPLETE (delegate to shared helper)
        result = self._complete_approved(pipeline_id)
        result["book_type"] = profile.get("book_type")
        result["confidence"] = confidence
        return result

    @staticmethod
    def _notify_pending_approval(book_name: str, confidence: float):
        """Send a macOS notification for books needing human approval."""
        if sys.platform != "darwin":
            return
        try:
            safe_name = book_name.replace('\\', '').replace('"', "'").replace("'", "\u2019")
            subprocess.Popen([
                "osascript", "-e",
                f'display notification "\\"{safe_name}\\" needs approval (confidence: {confidence:.0%})" '
                f'with title "Agentic Pipeline" subtitle "Book Pending Approval" '
                f'sound name "Purr"',
            ])
        except Exception as e:
            logger.debug("Notification failed (non-critical): %s", e)

    def _complete_approved(self, pipeline_id: str) -> dict:
        """Complete an approved book by running embedding and marking complete.

        Delegates to the shared _complete_approved in approval.actions.
        """
        from agentic_pipeline.approval.actions import _complete_approved

        record = self.repo.get(pipeline_id)
        if not record:
            raise ProcessingError(f"Pipeline not found: {pipeline_id}")

        result = _complete_approved(self.config.db_path, pipeline_id, record)
        result["pipeline_id"] = pipeline_id
        return result

    def _retry_one(self, book: dict) -> dict:
        """Retry a single book in NEEDS_RETRY state."""
        retry_count = book.get("retry_count", 0)

        if retry_count >= self.config.max_retry_attempts:
            self.logger.error(
                book["id"],
                "MaxRetriesExceeded",
                f"Exceeded {self.config.max_retry_attempts} retries",
            )
            self.repo.update_state(
                book["id"],
                PipelineState.REJECTED,
                error_details={"reason": "max_retries_exceeded"},
            )
            return {
                "pipeline_id": book["id"],
                "state": PipelineState.REJECTED.value,
                "reason": "max_retries_exceeded",
            }

        new_count = self.repo.increment_retry_count(book["id"])
        self.logger.retry_scheduled(book["id"], new_count)

        try:
            return self._process_book(
                book["id"], book["source_path"], book["content_hash"]
            )
        except Exception as e:
            self.logger.error(book["id"], type(e).__name__, str(e))
            return {
                "pipeline_id": book["id"],
                "state": PipelineState.NEEDS_RETRY.value,
                "error": str(e),
            }

    def _scan_watch_dir(self) -> int:
        """Scan watch directory for new book files and queue them."""
        if not self.config.watch_dir:
            return 0

        watch_path = Path(self.config.watch_dir)
        if not watch_path.is_dir():
            return 0

        extensions = ("*.epub", "*.pdf")
        detected = 0

        for ext in extensions:
            for book_path in watch_path.rglob(ext):
                # Skip files in processed directory
                if self.config.processed_dir and book_path.is_relative_to(self.config.processed_dir):
                    continue
                path_str = str(book_path)
                if path_str in self._seen_paths:
                    continue
                try:
                    content_hash = self._compute_hash(path_str)
                    if self._check_idempotency(content_hash):
                        self._seen_paths.add(path_str)
                        continue  # Already in pipeline
                    self.repo.create(path_str, content_hash)
                    self._seen_paths.add(path_str)
                    self.logger.processing_started("detected", path_str)
                    detected += 1
                except sqlite3.IntegrityError:
                    self._seen_paths.add(path_str)
                    continue  # Race: another cycle already created this record
                except Exception as e:
                    self.logger.error("scan", type(e).__name__, str(e))

        return detected

    def run_worker(self):
        """Run as queue worker, processing books continuously."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.logger.worker_started()

        while not self.shutdown_requested:
            # Priority 1: Complete approved books
            approved = self.repo.find_by_state(PipelineState.APPROVED, limit=1)
            if approved:
                try:
                    self._complete_approved(approved[0]["id"])
                except Exception as e:
                    self.logger.error(approved[0]["id"], type(e).__name__, str(e))
                    self.repo.update_state(approved[0]["id"], PipelineState.NEEDS_RETRY)
                continue

            # Priority 2: Process new books
            pending = self.repo.find_by_state(PipelineState.DETECTED, limit=1)
            if pending:
                try:
                    self._process_book(
                        pending[0]["id"],
                        pending[0]["source_path"],
                        pending[0]["content_hash"],
                    )
                except Exception as e:
                    self.logger.error(pending[0]["id"], type(e).__name__, str(e))
                    self.repo.update_state(pending[0]["id"], PipelineState.NEEDS_RETRY)
                continue

            # Priority 3: Retry failed
            retryable = self.repo.find_by_state(PipelineState.NEEDS_RETRY, limit=1)
            if retryable:
                try:
                    self._retry_one(retryable[0])
                except Exception as e:
                    self.logger.error(retryable[0]["id"], type(e).__name__, str(e))
                continue

            # Priority 4: Scan watch directory for new books
            if self.config.watch_dir:
                detected = self._scan_watch_dir()
                if detected > 0:
                    self.logger.state_transition("watch", "scanning", f"detected:{detected}")
                    continue

            time.sleep(self.config.worker_poll_interval)

        self.logger.worker_stopped()

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal gracefully."""
        self.logger.state_transition("worker", "running", "shutting_down")
        self.shutdown_requested = True

    def retry_failed(self) -> list[dict]:
        """Retry books in NEEDS_RETRY state."""
        results = []
        retryable = self.repo.find_by_state(PipelineState.NEEDS_RETRY)

        for book in retryable:
            result = self._retry_one(book)
            results.append(result)

        return results
