# Phase 3: Pipeline Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline orchestrator that moves books through classification, processing, and embedding automatically, with confidence-based routing and graceful error handling.

**Architecture:** Orchestrator class with two modes (one-shot, queue worker). Calls existing book-ingestion-python via subprocess. Uses shared SQLite database. Graceful shutdown, configurable timeouts, structured logging.

**Tech Stack:** Python 3.12, subprocess, signal handling, click (CLI), pytest

---

## Prerequisites

- Working directory: `/Users/taylorstephens/_Projects/book-mcp-server`
- Phase 2 complete (ClassifierAgent working)
- book-ingestion-python at `/Users/taylorstephens/_Projects/book-ingestion-python`
- Virtual environment: `source venv/bin/activate`

---

## Task 1: Configuration Module

**Files:**
- Create: `agentic_pipeline/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for orchestrator configuration."""

import pytest
import os


def test_config_loads_defaults():
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig()

    assert config.processing_timeout == 600
    assert config.embedding_timeout == 300
    assert config.confidence_threshold == 0.7
    assert config.worker_poll_interval == 5
    assert config.max_retry_attempts == 3


def test_config_from_env(monkeypatch):
    from agentic_pipeline.config import OrchestratorConfig

    monkeypatch.setenv("PROCESSING_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")

    config = OrchestratorConfig.from_env()

    assert config.processing_timeout == 900
    assert config.confidence_threshold == 0.8


def test_config_book_ingestion_path():
    from agentic_pipeline.config import OrchestratorConfig
    from pathlib import Path

    config = OrchestratorConfig()

    assert isinstance(config.book_ingestion_path, Path)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_config.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.config'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/config.py
"""Configuration for the pipeline orchestrator."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Paths
    db_path: Path = field(default_factory=lambda: Path(
        os.environ.get("AGENTIC_PIPELINE_DB", "data/library.db")
    ))
    book_ingestion_path: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "BOOK_INGESTION_PATH",
            "/Users/taylorstephens/_Projects/book-ingestion-python"
        )
    ))

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
        return cls(
            db_path=Path(os.environ.get("AGENTIC_PIPELINE_DB", "data/library.db")),
            book_ingestion_path=Path(os.environ.get(
                "BOOK_INGESTION_PATH",
                "/Users/taylorstephens/_Projects/book-ingestion-python"
            )),
            processing_timeout=int(os.environ.get("PROCESSING_TIMEOUT_SECONDS", 600)),
            embedding_timeout=int(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", 300)),
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", 0.7)),
            worker_poll_interval=int(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", 5)),
            max_retry_attempts=int(os.environ.get("MAX_RETRY_ATTEMPTS", 3)),
        )
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_config.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/config.py tests/test_config.py
git commit -m "feat: add OrchestratorConfig for pipeline settings"
```

---

## Task 2: Error Types

**Files:**
- Create: `agentic_pipeline/orchestrator/__init__.py`
- Create: `agentic_pipeline/orchestrator/errors.py`
- Create: `tests/test_orchestrator_errors.py`

**Step 1: Create orchestrator package**

Run: `mkdir -p agentic_pipeline/orchestrator && touch agentic_pipeline/orchestrator/__init__.py`

**Step 2: Write the failing test**

```python
# tests/test_orchestrator_errors.py
"""Tests for orchestrator error types."""

import pytest


def test_processing_error():
    from agentic_pipeline.orchestrator.errors import ProcessingError

    error = ProcessingError("Failed to extract text", exit_code=1)

    assert str(error) == "Failed to extract text"
    assert error.exit_code == 1


def test_embedding_error():
    from agentic_pipeline.orchestrator.errors import EmbeddingError

    error = EmbeddingError("Model not found")

    assert str(error) == "Model not found"


def test_timeout_error():
    from agentic_pipeline.orchestrator.errors import PipelineTimeoutError

    error = PipelineTimeoutError("Processing exceeded 600s", timeout=600)

    assert error.timeout == 600
```

**Step 3: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator_errors.py -v`

Expected: FAIL

**Step 4: Write minimal implementation**

```python
# agentic_pipeline/orchestrator/errors.py
"""Custom error types for the orchestrator."""


class OrchestratorError(Exception):
    """Base error for orchestrator operations."""
    pass


class ProcessingError(OrchestratorError):
    """Error during book processing (text extraction, cleaning)."""

    def __init__(self, message: str, exit_code: int = None, stderr: str = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class EmbeddingError(OrchestratorError):
    """Error during embedding generation."""

    def __init__(self, message: str, exit_code: int = None):
        super().__init__(message)
        self.exit_code = exit_code


class PipelineTimeoutError(OrchestratorError):
    """Operation exceeded timeout."""

    def __init__(self, message: str, timeout: int = None):
        super().__init__(message)
        self.timeout = timeout


class IdempotencyError(OrchestratorError):
    """Book already processed or in progress."""

    def __init__(self, message: str, existing_state: str = None):
        super().__init__(message)
        self.existing_state = existing_state
```

**Step 5: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator_errors.py -v`

Expected: 3 passed

**Step 6: Commit**

```bash
git add agentic_pipeline/orchestrator/ tests/test_orchestrator_errors.py
git commit -m "feat: add orchestrator error types"
```

---

## Task 3: Pipeline Logger

**Files:**
- Create: `agentic_pipeline/orchestrator/logging.py`
- Create: `tests/test_orchestrator_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator_logging.py
"""Tests for pipeline logging."""

import pytest
import json
import logging


def test_pipeline_logger_state_transition(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.INFO):
        logger.state_transition("abc123", "CLASSIFYING", "SELECTING_STRATEGY")

    assert "state_transition" in caplog.text
    assert "abc123" in caplog.text
    assert "CLASSIFYING" in caplog.text


def test_pipeline_logger_error(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.ERROR):
        logger.error("abc123", "ProcessingError", "Failed to extract text")

    assert "error" in caplog.text
    assert "ProcessingError" in caplog.text


def test_pipeline_logger_json_format(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.INFO):
        logger.processing_started("abc123", "/path/to/book.epub")

    # Should be valid JSON
    log_line = caplog.records[0].message
    data = json.loads(log_line)
    assert data["event"] == "processing_started"
    assert data["pipeline_id"] == "abc123"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator_logging.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator_logging.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/orchestrator/logging.py tests/test_orchestrator_logging.py
git commit -m "feat: add structured PipelineLogger"
```

---

## Task 4: Repository Updates

**Files:**
- Modify: `agentic_pipeline/db/pipelines.py`
- Create: `tests/test_pipelines_extended.py`

**Step 1: Write the failing test**

```python
# tests/test_pipelines_extended.py
"""Extended tests for pipeline repository."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_find_by_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create pipelines in different states
    id1 = repo.create("/book1.epub", "hash1")
    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.CLASSIFYING)

    detected = repo.find_by_state(PipelineState.DETECTED)

    assert len(detected) == 1
    assert detected[0]["id"] == id1


def test_increment_retry_count(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")

    count1 = repo.increment_retry_count(pid)
    count2 = repo.increment_retry_count(pid)

    assert count1 == 1
    assert count2 == 2


def test_find_by_state_with_limit(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create 5 pipelines
    for i in range(5):
        repo.create(f"/book{i}.epub", f"hash{i}")

    # Get only 2
    results = repo.find_by_state(PipelineState.DETECTED, limit=2)

    assert len(results) == 2
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_pipelines_extended.py -v`

Expected: FAIL (find_by_state doesn't exist)

**Step 3: Add methods to PipelineRepository**

Add to `agentic_pipeline/db/pipelines.py` (after existing methods):

```python
def find_by_state(
    self,
    state: PipelineState,
    limit: int = None
) -> list[dict]:
    """Find pipelines in a specific state."""
    conn = self._connect()
    cursor = conn.cursor()

    query = """
        SELECT * FROM processing_pipelines
        WHERE state = ?
        ORDER BY priority ASC, created_at ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, (state.value,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def increment_retry_count(self, pipeline_id: str) -> int:
    """Increment retry count and return new value."""
    conn = self._connect()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE processing_pipelines
        SET retry_count = retry_count + 1, updated_at = ?
        WHERE id = ?
    """, (datetime.utcnow().isoformat(), pipeline_id))

    cursor.execute(
        "SELECT retry_count FROM processing_pipelines WHERE id = ?",
        (pipeline_id,)
    )
    new_count = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    return new_count
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_pipelines_extended.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/db/pipelines.py tests/test_pipelines_extended.py
git commit -m "feat: add find_by_state and increment_retry_count to repository"
```

---

## Task 5: Orchestrator Core Class

**Files:**
- Create: `agentic_pipeline/orchestrator/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
"""Tests for the Pipeline Orchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def config(db_path):
    from agentic_pipeline.config import OrchestratorConfig

    return OrchestratorConfig(
        db_path=db_path,
        book_ingestion_path=Path("/mock/path"),
        processing_timeout=10,
        embedding_timeout=5,
    )


def test_orchestrator_initializes(config):
    from agentic_pipeline.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    assert orchestrator.config == config
    assert orchestrator.shutdown_requested == False


def test_orchestrator_idempotency_skips_complete(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Create a completed pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.HASHING)
    repo.update_state(pid, PipelineState.COMPLETE)

    orchestrator = Orchestrator(config)

    # Mock file hashing to return same hash
    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "already" in result["reason"].lower()


def test_orchestrator_idempotency_skips_in_progress(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Create an in-progress pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.HASHING)
    repo.update_state(pid, PipelineState.PROCESSING)

    orchestrator = Orchestrator(config)

    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "in progress" in result["reason"].lower()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Update orchestrator __init__.py**

```python
# agentic_pipeline/orchestrator/__init__.py
"""Pipeline Orchestrator package."""

from agentic_pipeline.orchestrator.orchestrator import Orchestrator
from agentic_pipeline.orchestrator.errors import (
    OrchestratorError,
    ProcessingError,
    EmbeddingError,
    PipelineTimeoutError,
    IdempotencyError,
)

__all__ = [
    "Orchestrator",
    "OrchestratorError",
    "ProcessingError",
    "EmbeddingError",
    "PipelineTimeoutError",
    "IdempotencyError",
]
```

**Step 5: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator.py -v`

Expected: 3 passed

**Step 6: Commit**

```bash
git add agentic_pipeline/orchestrator/ tests/test_orchestrator.py
git commit -m "feat: add Orchestrator core with idempotency checks"
```

---

## Task 6: State Processors

**Files:**
- Modify: `agentic_pipeline/orchestrator/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: Add state processing tests**

Add to `tests/test_orchestrator.py`:

```python
def test_orchestrator_classifies_book(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType
    from unittest.mock import MagicMock
    import subprocess

    orchestrator = Orchestrator(config)

    # Mock classifier
    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test"
    )
    orchestrator.classifier = MagicMock()
    orchestrator.classifier.classify.return_value = mock_profile

    # Mock subprocess for processing
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Mock file reading for text extraction
        with patch('builtins.open', MagicMock()):
            with patch.object(orchestrator, '_extract_sample', return_value="Chapter 1..."):
                with patch.object(orchestrator, '_compute_hash', return_value="newhash"):
                    result = orchestrator.process_one("/book.epub")

    # Should have called classifier
    orchestrator.classifier.classify.assert_called_once()


def test_orchestrator_handles_processing_timeout(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.pipeline.states import PipelineState
    import subprocess

    config.processing_timeout = 1  # 1 second timeout

    orchestrator = Orchestrator(config)

    # Mock to raise timeout
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 1)

        with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
            with patch.object(orchestrator, '_extract_sample', return_value="text"):
                with patch.object(orchestrator, '_run_classifier', return_value={}):
                    result = orchestrator.process_one("/book.epub")

    assert result["state"] == PipelineState.NEEDS_RETRY.value


def test_orchestrator_auto_approves_high_confidence(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType
    from unittest.mock import MagicMock
    import subprocess

    orchestrator = Orchestrator(config)

    # High confidence profile
    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,  # Above 0.7 threshold
        suggested_tags=["python"],
        reasoning="Test"
    )
    orchestrator.classifier = MagicMock()
    orchestrator.classifier.classify.return_value = mock_profile

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
            with patch.object(orchestrator, '_extract_sample', return_value="text"):
                result = orchestrator.process_one("/book.epub")

    # Check that it was auto-approved
    repo = PipelineRepository(db_path)
    pipeline = repo.get(result["pipeline_id"])
    assert pipeline["approved_by"] == "auto:high_confidence"
```

**Step 2: Implement state processing**

Update `agentic_pipeline/orchestrator/orchestrator.py`:

```python
# Add imports at top
import subprocess
from agentic_pipeline.agents.classifier import ClassifierAgent
from agentic_pipeline.pipeline.strategy import StrategySelector
from agentic_pipeline.orchestrator.errors import ProcessingError, EmbeddingError, PipelineTimeoutError

# Add to __init__:
def __init__(self, config: OrchestratorConfig):
    self.config = config
    self.repo = PipelineRepository(config.db_path)
    self.logger = PipelineLogger()
    self.classifier = ClassifierAgent(config.db_path)
    self.strategy_selector = StrategySelector()
    self.shutdown_requested = False

# Add these methods:
def _extract_sample(self, book_path: str, max_chars: int = 40000) -> str:
    """Extract text sample from book for classification."""
    # For now, just read the file if it's text
    # In production, this would use the book-ingestion converters
    path = Path(book_path)
    try:
        return path.read_text()[:max_chars]
    except UnicodeDecodeError:
        # Binary file - would need conversion
        return f"[Binary file: {path.name}]"

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
        result = subprocess.run(
            ["python", "-m", "src.cli", "process", book_path],
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
        result = subprocess.run(
            ["python", "-m", "scripts.generate_embeddings", "--book-hash", content_hash],
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
```

**Step 3: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator.py -v`

Expected: All tests pass

**Step 4: Commit**

```bash
git add agentic_pipeline/orchestrator/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: implement state processing in orchestrator"
```

---

## Task 7: Queue Worker with Graceful Shutdown

**Files:**
- Modify: `agentic_pipeline/orchestrator/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**Step 1: Add worker tests**

Add to `tests/test_orchestrator.py`:

```python
def test_orchestrator_worker_processes_queue(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    import threading

    repo = PipelineRepository(db_path)

    # Create a book in DETECTED state
    pid = repo.create("/book.epub", "hash123")

    orchestrator = Orchestrator(config)

    # Mock processing to just mark complete
    def mock_process(pipeline_id, book_path, content_hash):
        repo.update_state(pipeline_id, PipelineState.COMPLETE)
        return {"pipeline_id": pipeline_id, "state": "complete"}

    orchestrator._process_book = mock_process

    # Run worker in thread, stop after one iteration
    def run_and_stop():
        time.sleep(0.1)
        orchestrator.shutdown_requested = True

    stopper = threading.Thread(target=run_and_stop)
    stopper.start()

    orchestrator.run_worker()

    stopper.join()

    # Book should be processed
    pipeline = repo.get(pid)
    assert pipeline["state"] == PipelineState.COMPLETE.value


def test_orchestrator_graceful_shutdown(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    import signal

    orchestrator = Orchestrator(config)

    # Simulate SIGINT
    orchestrator._handle_shutdown(signal.SIGINT, None)

    assert orchestrator.shutdown_requested == True
```

**Step 2: Implement worker**

Add to `agentic_pipeline/orchestrator/orchestrator.py`:

```python
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
```

**Step 3: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/test_orchestrator.py -v`

Expected: All tests pass

**Step 4: Commit**

```bash
git add agentic_pipeline/orchestrator/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add queue worker with graceful shutdown"
```

---

## Task 8: CLI Commands

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Create: `tests/test_cli_orchestrator.py`

**Step 1: Write failing tests**

```python
# tests/test_cli_orchestrator.py
"""Tests for orchestrator CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_process_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["process", "--help"])

    assert result.exit_code == 0
    assert "process" in result.output.lower()


def test_worker_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["worker", "--help"])

    assert result.exit_code == 0


def test_retry_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["retry", "--help"])

    assert result.exit_code == 0


def test_status_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])

    assert result.exit_code == 0
```

**Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_cli_orchestrator.py -v`

Expected: FAIL

**Step 3: Add CLI commands**

Add to `agentic_pipeline/cli.py`:

```python
@main.command()
@click.argument("book_path", type=click.Path(exists=True))
def process(book_path: str):
    """Process a single book through the pipeline."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    console.print(f"[blue]Processing: {book_path}[/blue]")

    result = orchestrator.process_one(book_path)

    if result.get("skipped"):
        console.print(f"[yellow]Skipped: {result['reason']}[/yellow]")
        return

    state = result.get("state", "unknown")
    if state == "complete":
        console.print(f"[green]Complete![/green]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "pending_approval":
        console.print(f"[yellow]Pending approval[/yellow]")
        console.print(f"  Type: {result.get('book_type')}")
        console.print(f"  Confidence: {result.get('confidence', 0):.0%}")
    elif state == "needs_retry":
        console.print(f"[red]Failed - queued for retry[/red]")
        console.print(f"  Error: {result.get('error')}")
    else:
        console.print(f"[dim]State: {state}[/dim]")

    console.print(f"  Pipeline ID: {result.get('pipeline_id')}")


@main.command()
def worker():
    """Run the queue worker (processes books continuously)."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    console.print("[blue]Starting worker... Press Ctrl+C to stop gracefully.[/blue]")
    orchestrator.run_worker()
    console.print("[green]Worker stopped.[/green]")


@main.command()
@click.option("--max-attempts", "-m", default=3, help="Max retry attempts before rejection")
def retry(max_attempts: int):
    """Retry books in NEEDS_RETRY state."""
    from .config import OrchestratorConfig
    from .orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    config.max_retry_attempts = max_attempts
    orchestrator = Orchestrator(config)

    console.print(f"[blue]Retrying failed books (max {max_attempts} attempts)...[/blue]")

    results = orchestrator.retry_failed()

    if not results:
        console.print("[yellow]No books to retry[/yellow]")
        return

    for result in results:
        pid = result.get("pipeline_id", "?")[:8]
        state = result.get("state")
        if state == "complete":
            console.print(f"  [green]{pid}...: Complete[/green]")
        elif state == "rejected":
            console.print(f"  [red]{pid}...: Rejected ({result.get('reason')})[/red]")
        else:
            console.print(f"  [yellow]{pid}...: {state}[/yellow]")


@main.command()
@click.argument("pipeline_id")
def status(pipeline_id: str):
    """Show status of a pipeline."""
    from .db.config import get_db_path
    from .db.pipelines import PipelineRepository
    import json

    db_path = get_db_path()
    repo = PipelineRepository(db_path)

    pipeline = repo.get(pipeline_id)

    if not pipeline:
        console.print(f"[red]Pipeline not found: {pipeline_id}[/red]")
        return

    console.print(f"\n[bold]Pipeline: {pipeline_id}[/bold]")
    console.print(f"  State: [cyan]{pipeline['state']}[/cyan]")
    console.print(f"  Source: {pipeline['source_path']}")

    if pipeline.get("book_profile"):
        profile = json.loads(pipeline["book_profile"]) if isinstance(pipeline["book_profile"], str) else pipeline["book_profile"]
        console.print(f"  Type: {profile.get('book_type')}")
        conf = profile.get("confidence", 0)
        conf_style = "green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red"
        console.print(f"  Confidence: [{conf_style}]{conf:.0%}[/{conf_style}]")

    console.print(f"  Retries: {pipeline.get('retry_count', 0)}")
    console.print(f"  Created: {pipeline.get('created_at')}")

    if pipeline.get("approved_by"):
        console.print(f"  Approved by: {pipeline['approved_by']}")
```

**Step 4: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/test_cli_orchestrator.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_orchestrator.py
git commit -m "feat: add process, worker, retry, status CLI commands"
```

---

## Task 9: MCP Tools

**Files:**
- Modify: `agentic_pipeline/mcp_server.py`
- Create: `tests/test_mcp_orchestrator.py`

**Step 1: Write failing tests**

```python
# tests/test_mcp_orchestrator.py
"""Tests for orchestrator MCP tools."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def db_path(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_process_book_tool_exists(db_path):
    from agentic_pipeline.mcp_server import process_book

    assert callable(process_book)


def test_get_pipeline_status_tool_exists(db_path):
    from agentic_pipeline.mcp_server import get_pipeline_status

    assert callable(get_pipeline_status)
```

**Step 2: Read existing MCP server**

Run: Read `agentic_pipeline/mcp_server.py` to understand the structure.

**Step 3: Add MCP tools**

Add to `agentic_pipeline/mcp_server.py`:

```python
@mcp.tool()
def process_book(path: str) -> dict:
    """
    Process a book through the pipeline.

    Args:
        path: Path to the book file (epub, pdf, etc.)

    Returns:
        Processing result with pipeline_id, state, book_type, and confidence
    """
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    result = orchestrator.process_one(path)
    return result


@mcp.tool()
def get_pipeline_status(pipeline_id: str) -> dict:
    """
    Get the status of a pipeline run.

    Args:
        pipeline_id: The pipeline ID to check

    Returns:
        Pipeline details including state, book_type, confidence, retries
    """
    from agentic_pipeline.db.config import get_db_path
    from agentic_pipeline.db.pipelines import PipelineRepository
    import json

    db_path = get_db_path()
    repo = PipelineRepository(db_path)

    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"error": f"Pipeline not found: {pipeline_id}"}

    result = {
        "pipeline_id": pipeline_id,
        "state": pipeline["state"],
        "source_path": pipeline["source_path"],
        "retry_count": pipeline.get("retry_count", 0),
        "created_at": pipeline.get("created_at"),
    }

    if pipeline.get("book_profile"):
        profile = json.loads(pipeline["book_profile"]) if isinstance(pipeline["book_profile"], str) else pipeline["book_profile"]
        result["book_type"] = profile.get("book_type")
        result["confidence"] = profile.get("confidence")
        result["suggested_tags"] = profile.get("suggested_tags")

    if pipeline.get("approved_by"):
        result["approved_by"] = pipeline["approved_by"]

    return result
```

**Step 4: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/test_mcp_orchestrator.py -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/mcp_server.py tests/test_mcp_orchestrator.py
git commit -m "feat: add process_book and get_pipeline_status MCP tools"
```

---

## Task 10: Integration Test & Final Verification

**Files:**
- Create: `tests/test_orchestrator_integration.py`

**Step 1: Write integration test**

```python
# tests/test_orchestrator_integration.py
"""Integration tests for the orchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_book(tmp_path):
    """Create a sample text file to process."""
    book = tmp_path / "sample.txt"
    book.write_text("""
    Chapter 1: Introduction to Python

    Python is a versatile programming language. In this tutorial,
    we will learn the basics of Python programming.

    def hello_world():
        print("Hello, World!")

    This function prints a greeting message.
    """)
    return str(book)


@pytest.fixture
def config(db_path):
    from agentic_pipeline.config import OrchestratorConfig

    return OrchestratorConfig(
        db_path=db_path,
        book_ingestion_path=Path("/mock/path"),
        processing_timeout=10,
        embedding_timeout=5,
        confidence_threshold=0.7,
    )


def test_full_pipeline_mocked(config, sample_book):
    """Test full pipeline with mocked subprocess calls."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType
    from agentic_pipeline.db.pipelines import PipelineRepository

    orchestrator = Orchestrator(config)

    # Mock classifier to return high confidence
    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python", "programming"],
        reasoning="Contains code examples"
    )

    with patch.object(orchestrator.classifier, 'classify', return_value=mock_profile):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            result = orchestrator.process_one(sample_book)

    assert result["state"] == "complete"
    assert result["book_type"] == "technical_tutorial"
    assert result["confidence"] == 0.9

    # Verify database state
    repo = PipelineRepository(config.db_path)
    pipeline = repo.get(result["pipeline_id"])
    assert pipeline["state"] == "complete"
    assert pipeline["approved_by"] == "auto:high_confidence"


def test_low_confidence_needs_approval(config, sample_book):
    """Test that low confidence books need approval."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    orchestrator = Orchestrator(config)

    # Low confidence profile
    mock_profile = BookProfile(
        book_type=BookType.UNKNOWN,
        confidence=0.5,
        suggested_tags=[],
        reasoning="Unclear structure"
    )

    with patch.object(orchestrator.classifier, 'classify', return_value=mock_profile):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = orchestrator.process_one(sample_book)

    assert result["state"] == "pending_approval"
    assert result["needs_review"] == True
```

**Step 2: Run all tests**

Run: `source venv/bin/activate && python -m pytest tests/ -v`

Expected: All tests pass

**Step 3: Verify CLI works**

Run: `source venv/bin/activate && python -m agentic_pipeline.cli process --help`

Expected: Shows help for process command

**Step 4: Final commit**

```bash
git add tests/test_orchestrator_integration.py
git commit -m "test: add orchestrator integration tests"
```

**Step 5: Summary commit**

```bash
git add -A
git commit -m "feat: complete Phase 3 - Pipeline Orchestrator

- OrchestratorConfig for environment-based configuration
- Custom error types (ProcessingError, EmbeddingError, etc.)
- Structured PipelineLogger with JSON output
- Orchestrator with process_one() and run_worker() modes
- State processing: hash → classify → process → validate → approve → embed
- Graceful shutdown on SIGINT/SIGTERM
- Retry logic with max attempts
- CLI commands: process, worker, retry, status
- MCP tools: process_book, get_pipeline_status
- Confidence-based auto-approval (≥70%)
- Idempotency checks (skip completed/in-progress)

Ready for integration testing with real books"
```

---

## Summary

Phase 3 delivers:

| Component | Status |
|-----------|--------|
| OrchestratorConfig | ✅ |
| Error types | ✅ |
| PipelineLogger | ✅ |
| Repository updates | ✅ |
| Orchestrator core | ✅ |
| State processors | ✅ |
| Queue worker | ✅ |
| CLI commands | ✅ |
| MCP tools | ✅ |
| Integration tests | ✅ |

**Total: 10 tasks, ~60 steps, ~10 commits**

**New files:**
- `agentic_pipeline/config.py`
- `agentic_pipeline/orchestrator/__init__.py`
- `agentic_pipeline/orchestrator/orchestrator.py`
- `agentic_pipeline/orchestrator/errors.py`
- `agentic_pipeline/orchestrator/logging.py`

**Modified files:**
- `agentic_pipeline/cli.py`
- `agentic_pipeline/db/pipelines.py`
- `agentic_pipeline/mcp_server.py`

Next: Phase 4 - File Watcher (optional) or real-world testing with actual books.
