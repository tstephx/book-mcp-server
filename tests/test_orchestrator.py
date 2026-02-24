# tests/test_orchestrator.py
"""Tests for the Pipeline Orchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from conftest import transition_to


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
    transition_to(repo, pid, PipelineState.COMPLETE)

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
    transition_to(repo, pid, PipelineState.PROCESSING)

    orchestrator = Orchestrator(config)

    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "in progress" in result["reason"].lower()


def test_orchestrator_idempotency_skips_failed(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Create a permanently failed pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    transition_to(repo, pid, PipelineState.FAILED)

    orchestrator = Orchestrator(config)

    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "permanently failed" in result["reason"].lower()


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

    mock_processing_result = {
        "book_id": "test-book-id",
        "quality_score": 85,
        "detection_confidence": 0.9,
        "detection_method": "mock",
        "needs_review": False,
        "warnings": [],
        "chapter_count": 10,
        "word_count": 50000,
        "llm_fallback_used": False,
    }

    from agentic_pipeline.validation import ValidationResult
    mock_validation = ValidationResult(passed=True, reasons=[], warnings=[], metrics={"chapter_count": 10})

    with patch.object(orchestrator, '_run_processing', return_value=mock_processing_result):
        with patch.object(orchestrator, '_run_embedding', return_value={"chapters_processed": 10}):
            with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
                with patch.object(orchestrator, '_extract_sample', return_value="text"):
                    with patch("agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate", return_value=mock_validation):
                        result = orchestrator.process_one("/book.epub")

    # Check that it was auto-approved
    repo = PipelineRepository(db_path)
    pipeline = repo.get(result["pipeline_id"])
    assert pipeline["approved_by"] == "auto:high_confidence"


def test_orchestrator_worker_processes_queue(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    import threading
    import time

    repo = PipelineRepository(db_path)

    # Create a book in DETECTED state
    pid = repo.create("/book.epub", "hash123")

    orchestrator = Orchestrator(config)

    # Mock processing to just mark complete
    def mock_process(pipeline_id, book_path, content_hash):
        transition_to(repo, pipeline_id, PipelineState.COMPLETE)
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


def test_scan_directory_finds_new_books(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    # Create fake book files
    (tmp_path / "book1.epub").write_bytes(b"fake epub 1")
    (tmp_path / "book2.pdf").write_bytes(b"fake pdf 2")
    (tmp_path / "readme.txt").write_bytes(b"not a book")

    config.watch_dir = tmp_path
    orchestrator = Orchestrator(config)
    detected = orchestrator._scan_watch_dir()

    assert detected == 2


def test_scan_directory_skips_already_queued(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    (tmp_path / "book1.epub").write_bytes(b"fake epub 1")

    config.watch_dir = tmp_path
    orchestrator = Orchestrator(config)

    # First scan detects the book
    assert orchestrator._scan_watch_dir() == 1
    # Second scan skips it (cached in _seen_paths, no re-hash needed)
    assert orchestrator._scan_watch_dir() == 0


def test_scan_directory_uses_seen_cache(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator
    from unittest.mock import patch

    (tmp_path / "book1.epub").write_bytes(b"fake epub 1")

    config.watch_dir = tmp_path
    orchestrator = Orchestrator(config)

    # First scan hashes and detects the book
    assert orchestrator._scan_watch_dir() == 1

    # Second scan should NOT call _compute_hash (path is cached)
    with patch.object(orchestrator, '_compute_hash') as mock_hash:
        assert orchestrator._scan_watch_dir() == 0
        mock_hash.assert_not_called()


def test_scan_directory_noop_when_no_watch_dir(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator

    config.watch_dir = None
    orchestrator = Orchestrator(config)
    assert orchestrator._scan_watch_dir() == 0


def test_scan_directory_noop_when_dir_missing(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    config.watch_dir = tmp_path / "nonexistent"
    orchestrator = Orchestrator(config)
    assert orchestrator._scan_watch_dir() == 0


def test_worker_scans_watch_dir(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    import threading
    import time

    (tmp_path / "newbook.epub").write_bytes(b"new book content")
    config.watch_dir = tmp_path
    config.worker_poll_interval = 0  # No delay for tests

    orchestrator = Orchestrator(config)

    # Mock _process_book to just mark complete
    repo = PipelineRepository(db_path)

    def mock_process(pid, path, hash):
        transition_to(repo, pid, PipelineState.COMPLETE)
        return {"pipeline_id": pid, "state": "complete"}

    orchestrator._process_book = mock_process

    def stop_later():
        time.sleep(0.3)
        orchestrator.shutdown_requested = True

    threading.Thread(target=stop_later).start()
    orchestrator.run_worker()

    # The new book should have been detected and processed
    pipelines = repo.find_by_state(PipelineState.COMPLETE)
    assert len(pipelines) >= 1


def test_retry_one_uses_failed_not_rejected_on_max_retries():
    """_retry_one transitions to FAILED (not REJECTED) when max retries exceeded."""
    from unittest.mock import MagicMock
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator
    from agentic_pipeline.pipeline.states import PipelineState

    config = MagicMock()
    config.max_retry_attempts = 2
    config.autonomy_mode = "supervised"

    repo = MagicMock()
    logger = MagicMock()

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = config
    orch.repo = repo
    orch.logger = logger

    book = {"id": "book-1", "retry_count": 2, "source_path": "/tmp/x.epub", "content_hash": "abc"}
    result = orch._retry_one(book)

    assert result["state"] == PipelineState.FAILED.value
    assert result["reason"] == "max_retries_exceeded"
    repo.update_state.assert_called_once_with(
        "book-1",
        PipelineState.FAILED,
        error_details={"reason": "max_retries_exceeded"},
    )
