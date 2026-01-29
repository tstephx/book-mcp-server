# agentic_pipeline/orchestrator/logging.py
"""Structured logging for the pipeline orchestrator."""

import json
import logging
from datetime import datetime, timezone


class PipelineLogger:
    """Structured JSON logger for pipeline events."""

    def __init__(self, name: str = "orchestrator"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _log(self, level: int, event: str, **kwargs):
        """Log a structured event."""
        data = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.logger.log(level, json.dumps(data))

    def state_transition(self, pipeline_id: str, from_state: str, to_state: str):
        """Log a state transition."""
        self._log(
            logging.INFO,
            "state_transition",
            pipeline_id=pipeline_id,
            from_state=from_state,
            to_state=to_state
        )

    def processing_started(self, pipeline_id: str, book_path: str):
        """Log processing start."""
        self._log(
            logging.INFO,
            "processing_started",
            pipeline_id=pipeline_id,
            book_path=book_path
        )

    def processing_complete(self, pipeline_id: str, duration_seconds: float):
        """Log processing completion."""
        self._log(
            logging.INFO,
            "processing_complete",
            pipeline_id=pipeline_id,
            duration_seconds=round(duration_seconds, 2)
        )

    def error(self, pipeline_id: str, error_type: str, message: str):
        """Log an error."""
        self._log(
            logging.ERROR,
            "error",
            pipeline_id=pipeline_id,
            error_type=error_type,
            message=message
        )

    def worker_started(self):
        """Log worker start."""
        self._log(logging.INFO, "worker_started")

    def worker_stopped(self):
        """Log worker stop."""
        self._log(logging.INFO, "worker_stopped")

    def retry_scheduled(self, pipeline_id: str, retry_count: int):
        """Log retry scheduling."""
        self._log(
            logging.WARNING,
            "retry_scheduled",
            pipeline_id=pipeline_id,
            retry_count=retry_count
        )
