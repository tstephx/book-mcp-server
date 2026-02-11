# agentic_pipeline/config.py
"""Configuration for the pipeline orchestrator."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.config import get_db_path


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Paths
    db_path: Path = field(default_factory=get_db_path)
    watch_dir: Optional[Path] = None

    # Timeouts (seconds)
    processing_timeout: int = 600  # 10 minutes
    embedding_timeout: int = 300   # 5 minutes

    # Thresholds
    confidence_threshold: float = 0.7

    # Worker settings
    worker_poll_interval: int = 5  # seconds
    max_retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Create configuration from environment variables."""
        watch_dir_str = os.environ.get("WATCH_DIR")
        return cls(
            db_path=get_db_path(),
            watch_dir=Path(watch_dir_str) if watch_dir_str else None,
            processing_timeout=int(os.environ.get("PROCESSING_TIMEOUT_SECONDS", 600)),
            embedding_timeout=int(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", 300)),
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", 0.7)),
            worker_poll_interval=int(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", 5)),
            max_retry_attempts=int(os.environ.get("MAX_RETRY_ATTEMPTS", 3)),
        )
