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
