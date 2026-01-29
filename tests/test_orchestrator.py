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
