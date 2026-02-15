# tests/test_validation_retry.py
"""Tests for validation retry with force_fallback on first failure."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call


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
    book.write_text("Chapter 1: Introduction\nSome content here.\n")
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


def _make_mock_profile():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    return BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test book",
    )


def _make_processing_result():
    return {
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


def test_first_validation_failure_retries(config, sample_book):
    """First validation fails, retry with force_fallback passes -> book completes."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    mock_profile = _make_mock_profile()
    mock_processing_result = _make_processing_result()

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 1 (minimum 7 required)"],
        warnings=[],
        metrics={"chapter_count": 1},
    )
    pass_validation = ValidationResult(
        passed=True,
        reasons=[],
        warnings=[],
        metrics={"chapter_count": 10},
    )

    # Validator fails first time, passes second time
    mock_validate = MagicMock(side_effect=[fail_validation, pass_validation])

    # Mock embedding for the complete flow
    mock_adapter_cls = MagicMock()
    mock_adapter_instance = mock_adapter_cls.return_value
    mock_embed_result = MagicMock(success=True, chapters_processed=10, error=None)
    mock_adapter_instance.generate_embeddings.return_value = mock_embed_result

    with patch.object(orchestrator.classifier, "classify", return_value=mock_profile):
        with patch.object(orchestrator, "_run_processing", return_value=mock_processing_result):
            with patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", mock_adapter_cls):
                with patch(
                    "agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate",
                    mock_validate,
                ):
                    result = orchestrator.process_one(sample_book)

    assert result["state"] == "complete"
    # Validator should have been called twice
    assert mock_validate.call_count == 2


def test_both_attempts_fail_rejects(config, sample_book):
    """Both validation attempts fail -> book rejected."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    mock_profile = _make_mock_profile()
    mock_processing_result = _make_processing_result()

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 1 (minimum 7 required)"],
        warnings=[],
        metrics={"chapter_count": 1},
    )

    with patch.object(orchestrator.classifier, "classify", return_value=mock_profile):
        with patch.object(orchestrator, "_run_processing", return_value=mock_processing_result):
            with patch(
                "agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate",
                return_value=fail_validation,
            ):
                result = orchestrator.process_one(sample_book)

    assert result["state"] == "rejected"
    assert "Too few chapters" in result["reason"]
    # Validator should have been called twice (initial + retry)
    # We used return_value so both calls return failure


def test_retry_uses_force_fallback(config, sample_book):
    """Verify _run_processing is called with force_fallback=True on retry."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    mock_profile = _make_mock_profile()
    mock_processing_result = _make_processing_result()

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters"],
        warnings=[],
        metrics={"chapter_count": 1},
    )
    pass_validation = ValidationResult(
        passed=True,
        reasons=[],
        warnings=[],
        metrics={"chapter_count": 10},
    )

    mock_validate = MagicMock(side_effect=[fail_validation, pass_validation])

    mock_adapter_cls = MagicMock()
    mock_adapter_instance = mock_adapter_cls.return_value
    mock_embed_result = MagicMock(success=True, chapters_processed=10, error=None)
    mock_adapter_instance.generate_embeddings.return_value = mock_embed_result

    mock_run_processing = MagicMock(return_value=mock_processing_result)

    with patch.object(orchestrator.classifier, "classify", return_value=mock_profile):
        with patch.object(orchestrator, "_run_processing", mock_run_processing):
            with patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", mock_adapter_cls):
                with patch(
                    "agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate",
                    mock_validate,
                ):
                    result = orchestrator.process_one(sample_book)

    assert result["state"] == "complete"

    # First call: normal processing (no force_fallback)
    assert mock_run_processing.call_count == 2
    first_call = mock_run_processing.call_args_list[0]
    assert first_call == call(sample_book, book_id=result["pipeline_id"])

    # Second call: retry with force_fallback=True
    second_call = mock_run_processing.call_args_list[1]
    assert second_call == call(sample_book, book_id=result["pipeline_id"], force_fallback=True)


def test_retry_processing_error_rejects_directly(config, sample_book):
    """ProcessingError during force_fallback retry rejects without NEEDS_RETRY."""
    import json as json_module
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.orchestrator.errors import ProcessingError
    from agentic_pipeline.validation import ValidationResult
    from agentic_pipeline.db.connection import get_pipeline_db

    orchestrator = Orchestrator(config)

    mock_profile = _make_mock_profile()
    mock_processing_result = _make_processing_result()

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 1 (minimum 7 required)"],
        warnings=[],
        metrics={"chapter_count": 1},
    )

    # First call succeeds, second call (retry) raises ProcessingError
    mock_run_processing = MagicMock(
        side_effect=[mock_processing_result, ProcessingError("Splitter crashed")]
    )

    with patch.object(orchestrator.classifier, "classify", return_value=mock_profile):
        with patch.object(orchestrator, "_run_processing", mock_run_processing):
            with patch(
                "agentic_pipeline.validation.extraction_validator.ExtractionValidator.validate",
                return_value=fail_validation,
            ):
                result = orchestrator.process_one(sample_book)

    assert result["state"] == "rejected"
    assert result["error"] == "Splitter crashed"
    assert mock_run_processing.call_count == 2

    # Verify state history records initial_validation and retry_error
    with get_pipeline_db(str(config.db_path)) as conn:
        row = conn.execute(
            "SELECT error_details FROM pipeline_state_history "
            "WHERE pipeline_id = ? AND to_state = 'rejected' ORDER BY rowid DESC LIMIT 1",
            (result["pipeline_id"],),
        ).fetchone()
    assert row is not None
    error_details = json_module.loads(row["error_details"])
    assert "initial_validation" in error_details
    assert error_details["retry_error"] == "Splitter crashed"
