# agentic_pipeline/orchestrator/orchestrator.py
"""Pipeline Orchestrator - coordinates book processing."""

import hashlib
import signal
import time
from pathlib import Path
from typing import Optional

from agentic_pipeline.config import OrchestratorConfig
from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState, TERMINAL_STATES
from agentic_pipeline.orchestrator.logging import PipelineLogger
from agentic_pipeline.orchestrator.errors import IdempotencyError


class Orchestrator:
    """Orchestrates book processing through the pipeline."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.repo = PipelineRepository(config.db_path)
        self.logger = PipelineLogger()
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
        """Internal: process book through all states."""
        # This will be implemented in Task 6
        # For now, just return a placeholder
        return {
            "pipeline_id": pipeline_id,
            "state": "pending_implementation",
        }
