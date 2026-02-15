# tests/test_orchestrator_integration.py
"""Integration tests for the orchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


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
        processing_timeout=10,
        embedding_timeout=5,
        confidence_threshold=0.7,
    )


def test_full_pipeline_mocked(config, sample_book):
    """Test full pipeline with mocked processing."""
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

    # _complete_approved delegates to approval.actions which lazy-imports
    # ProcessingAdapter, so we patch at the source module.
    mock_adapter_cls = MagicMock()
    mock_adapter_instance = mock_adapter_cls.return_value
    mock_embed_result = MagicMock(success=True, chapters_processed=10, error=None)
    mock_adapter_instance.generate_embeddings.return_value = mock_embed_result

    from agentic_pipeline.validation import ValidationResult
    mock_validation = ValidationResult(passed=True, reasons=[], warnings=[], metrics={"chapter_count": 10})

    with patch.object(orchestrator.classifier, 'classify', return_value=mock_profile):
        with patch.object(orchestrator, '_run_processing', return_value=mock_processing_result):
            with patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", mock_adapter_cls):
                with patch("agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate", return_value=mock_validation):
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

    mock_processing_result = {
        "book_id": "test-book-id",
        "quality_score": 40,
        "detection_confidence": 0.5,
        "detection_method": "mock",
        "needs_review": True,
        "warnings": [],
        "chapter_count": 3,
        "word_count": 10000,
        "llm_fallback_used": False,
    }

    from agentic_pipeline.validation import ValidationResult
    mock_validation = ValidationResult(passed=True, reasons=[], warnings=[], metrics={"chapter_count": 3})

    with patch.object(orchestrator.classifier, 'classify', return_value=mock_profile):
        with patch.object(orchestrator, '_run_processing', return_value=mock_processing_result):
            with patch("agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate", return_value=mock_validation):
                result = orchestrator.process_one(sample_book)

    assert result["state"] == "pending_approval"
    assert result["needs_review"] == True


def test_validation_failure_rejects_book(config, sample_book):
    """Books failing extraction quality checks are rejected after retry with force_fallback."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType
    from agentic_pipeline.db.pipelines import PipelineRepository

    orchestrator = Orchestrator(config)

    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test book"
    )

    mock_processing_result = {
        "book_id": "test-book-id",
        "quality_score": 85,
        "detection_confidence": 0.9,
        "detection_method": "mock",
        "needs_review": False,
        "warnings": [],
        "chapter_count": 1,
        "word_count": 500,
        "llm_fallback_used": False,
    }

    from agentic_pipeline.validation import ValidationResult
    mock_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 1 (minimum 7 required)", "Total word count too low: 500 (minimum 5,000 required)"],
        warnings=[],
        metrics={"chapter_count": 1, "total_words": 500},
    )

    # Validation always fails (return_value), so both initial and retry fail -> rejected
    with patch.object(orchestrator.classifier, 'classify', return_value=mock_profile):
        with patch.object(orchestrator, '_run_processing', return_value=mock_processing_result):
            with patch("agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate", return_value=mock_validation):
                result = orchestrator.process_one(sample_book)

    assert result["state"] == "rejected"
    assert "Too few chapters" in result["reason"]
    assert "Total word count too low" in result["reason"]
    assert result["metrics"]["chapter_count"] == 1

    # Verify DB state
    repo = PipelineRepository(config.db_path)
    pipeline = repo.get(result["pipeline_id"])
    assert pipeline["state"] == "rejected"
