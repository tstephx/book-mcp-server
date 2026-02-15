"""Tests for force_fallback parameter threading through ProcessingAdapter."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_app():
    """Create a mock BookIngestionApp with a process() method."""
    app = MagicMock()
    app.process.return_value = MagicMock(
        success=True,
        book_id="test-book",
        pipeline_result=MagicMock(
            quality_report=MagicMock(quality_score=80),
            detection_confidence=0.9,
            detection_method="toc",
            needs_review=False,
            warnings=[],
            chapters=[
                {"word_count": 5000},
                {"word_count": 6000},
            ],
        ),
        llm_fallback_used=False,
    )
    return app


@patch("agentic_pipeline.adapters.processing_adapter.BookIngestionApp")
@patch("agentic_pipeline.adapters.processing_adapter.LLMFallbackAdapter")
def test_process_book_passes_force_fallback_true(mock_llm_cls, mock_app_cls, mock_app):
    """ProcessingAdapter.process_book(force_fallback=True) passes the flag to BookIngestionApp.process()."""
    mock_app_cls.create.return_value = mock_app

    from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

    adapter = ProcessingAdapter(db_path="/tmp/test.db")
    adapter.process_book(
        book_path="/tmp/test.epub",
        book_id="test-book",
        force_fallback=True,
    )

    mock_app.process.assert_called_once_with(
        file_path="/tmp/test.epub",
        title=None,
        author=None,
        book_id="test-book",
        save_to_storage=True,
        force_fallback=True,
    )


@patch("agentic_pipeline.adapters.processing_adapter.BookIngestionApp")
@patch("agentic_pipeline.adapters.processing_adapter.LLMFallbackAdapter")
def test_process_book_defaults_force_fallback_false(mock_llm_cls, mock_app_cls, mock_app):
    """ProcessingAdapter.process_book() without force_fallback passes False by default."""
    mock_app_cls.create.return_value = mock_app

    from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

    adapter = ProcessingAdapter(db_path="/tmp/test.db")
    adapter.process_book(
        book_path="/tmp/test.epub",
        book_id="test-book",
    )

    mock_app.process.assert_called_once_with(
        file_path="/tmp/test.epub",
        title=None,
        author=None,
        book_id="test-book",
        save_to_storage=True,
        force_fallback=False,
    )
