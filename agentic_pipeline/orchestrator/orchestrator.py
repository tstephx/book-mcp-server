# agentic_pipeline/orchestrator/orchestrator.py
"""Pipeline Orchestrator - coordinates book processing."""

import hashlib
import signal
import subprocess
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


class Orchestrator:
    """Orchestrates book processing through the pipeline."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.repo = PipelineRepository(config.db_path)
        self.logger = PipelineLogger()
        self.classifier = ClassifierAgent(config.db_path)
        self.strategy_selector = StrategySelector()
        self.shutdown_requested = False

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

    def _run_processing(self, book_path: str) -> None:
        """Run book-ingestion processing via subprocess."""
        try:
            # Use the venv python from book-ingestion-python
            venv_python = self.config.book_ingestion_path / "venv" / "bin" / "python"
            python_cmd = str(venv_python) if venv_python.exists() else "python"

            result = subprocess.run(
                [python_cmd, "-m", "src.cli", "process", book_path],
                cwd=str(self.config.book_ingestion_path),
                timeout=self.config.processing_timeout,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise ProcessingError(
                    f"Processing failed: {result.stderr}",
                    exit_code=result.returncode,
                    stderr=result.stderr
                )
        except subprocess.TimeoutExpired:
            raise PipelineTimeoutError(
                f"Processing exceeded {self.config.processing_timeout}s",
                timeout=self.config.processing_timeout
            )

    def _run_embedding(self, content_hash: str) -> None:
        """Run embedding generation via subprocess."""
        try:
            # Use the venv python from book-ingestion-python
            venv_python = self.config.book_ingestion_path / "venv" / "bin" / "python"
            python_cmd = str(venv_python) if venv_python.exists() else "python"

            # The generate_embeddings script processes all chapters without embeddings
            result = subprocess.run(
                [python_cmd, "scripts/generate_embeddings.py"],
                cwd=str(self.config.book_ingestion_path),
                timeout=self.config.embedding_timeout,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise EmbeddingError(
                    f"Embedding failed: {result.stderr}",
                    exit_code=result.returncode
                )
        except subprocess.TimeoutExpired:
            raise PipelineTimeoutError(
                f"Embedding exceeded {self.config.embedding_timeout}s",
                timeout=self.config.embedding_timeout
            )

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
            self._run_processing(book_path)
        except (ProcessingError, PipelineTimeoutError) as e:
            self.logger.error(pipeline_id, type(e).__name__, str(e))
            self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
            return {"pipeline_id": pipeline_id, "state": PipelineState.NEEDS_RETRY.value, "error": str(e)}

        # VALIDATING
        self._transition(pipeline_id, PipelineState.VALIDATING)
        # TODO: Add actual validation logic

        # APPROVAL ROUTING
        confidence = profile.get("confidence", 0)
        if confidence >= self.config.confidence_threshold:
            # Auto-approve
            self.repo.mark_approved(pipeline_id, approved_by="auto:high_confidence", confidence=confidence)
            self._transition(pipeline_id, PipelineState.APPROVED)
        else:
            # Needs human review
            self._transition(pipeline_id, PipelineState.PENDING_APPROVAL)
            return {
                "pipeline_id": pipeline_id,
                "state": PipelineState.PENDING_APPROVAL.value,
                "book_type": profile.get("book_type"),
                "confidence": confidence,
                "needs_review": True
            }

        # EMBEDDING
        self._transition(pipeline_id, PipelineState.EMBEDDING)
        try:
            self._run_embedding(content_hash)
        except (EmbeddingError, PipelineTimeoutError) as e:
            self.logger.error(pipeline_id, type(e).__name__, str(e))
            self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
            return {"pipeline_id": pipeline_id, "state": PipelineState.NEEDS_RETRY.value, "error": str(e)}

        # COMPLETE
        self._transition(pipeline_id, PipelineState.COMPLETE)

        return {
            "pipeline_id": pipeline_id,
            "state": PipelineState.COMPLETE.value,
            "book_type": profile.get("book_type"),
            "confidence": confidence
        }

    def run_worker(self):
        """Run as queue worker, processing books continuously."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.logger.worker_started()

        while not self.shutdown_requested:
            # Find next book to process
            pending = self.repo.find_by_state(PipelineState.DETECTED, limit=1)

            if not pending:
                time.sleep(self.config.worker_poll_interval)
                continue

            book = pending[0]
            try:
                self._process_book(book["id"], book["source_path"], book["content_hash"])
            except Exception as e:
                self.logger.error(book["id"], type(e).__name__, str(e))
                self.repo.update_state(book["id"], PipelineState.NEEDS_RETRY)

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
            retry_count = book.get("retry_count", 0)

            if retry_count >= self.config.max_retry_attempts:
                self.logger.error(
                    book["id"],
                    "MaxRetriesExceeded",
                    f"Exceeded {self.config.max_retry_attempts} retries"
                )
                self.repo.update_state(
                    book["id"],
                    PipelineState.REJECTED,
                    error_details={"reason": "max_retries_exceeded"}
                )
                results.append({
                    "pipeline_id": book["id"],
                    "state": PipelineState.REJECTED.value,
                    "reason": "max_retries_exceeded"
                })
                continue

            # Increment retry count
            new_count = self.repo.increment_retry_count(book["id"])
            self.logger.retry_scheduled(book["id"], new_count)

            # Attempt reprocessing
            try:
                result = self._process_book(
                    book["id"],
                    book["source_path"],
                    book["content_hash"]
                )
                results.append(result)
            except Exception as e:
                self.logger.error(book["id"], type(e).__name__, str(e))
                results.append({
                    "pipeline_id": book["id"],
                    "state": PipelineState.NEEDS_RETRY.value,
                    "error": str(e)
                })

        return results
