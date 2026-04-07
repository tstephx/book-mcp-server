"""Integration tests for ProcessingAdapter."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

pytest.importorskip("book_ingestion", reason="book_ingestion not installed (CI)")

from agentic_pipeline.adapters.processing_adapter import (
    ProcessingAdapter,
    ProcessingResult,
    EmbeddingResult,
)


def _make_mock_app(mock_result):
    """Return a mock BookIngestionApp whose .process() returns mock_result."""
    mock_app = MagicMock()
    mock_app.process.return_value = mock_result
    return mock_app


def _make_success_result(book_id="test-123", quality_score=85, chapters=None):
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.book_id = book_id
    mock_result.pipeline_result = MagicMock()
    mock_result.pipeline_result.quality_report.quality_score = quality_score
    mock_result.pipeline_result.detection_confidence = 0.9
    mock_result.pipeline_result.detection_method = "toc"
    mock_result.pipeline_result.needs_review = False
    mock_result.pipeline_result.warnings = []
    mock_result.pipeline_result.chapters = chapters or [{"word_count": 1000}]
    mock_result.llm_fallback_used = False
    return mock_result


class TestProcessingAdapterInit:
    """Tests for ProcessingAdapter initialization."""

    def test_creates_with_defaults(self, tmp_path):
        """Adapter initializes with default settings."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        assert adapter.db_path == db_path
        assert adapter.output_dir == tmp_path / "processed"

    def test_creates_with_custom_output_dir(self, tmp_path):
        """Adapter respects custom output directory."""
        db_path = tmp_path / "test.db"
        output_dir = tmp_path / "custom_output"

        adapter = ProcessingAdapter(db_path=db_path, output_dir=output_dir)

        assert adapter.output_dir == output_dir

    def test_stores_llm_fallback_config(self, tmp_path):
        """LLM fallback config is stored for per-call use."""
        db_path = tmp_path / "test.db"

        adapter = ProcessingAdapter(
            db_path=db_path,
            enable_llm_fallback=True,
            llm_fallback_threshold=0.6,
        )

        assert adapter._enable_llm_fallback is True
        assert adapter._llm_fallback_threshold == 0.6


class TestProcessingAdapterProcessBook:
    """Tests for ProcessingAdapter.process_book()."""

    def test_returns_processing_result(self, tmp_path):
        """process_book returns a ProcessingResult dataclass."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        mock_result = _make_success_result()
        mock_app = _make_mock_app(mock_result)

        with patch(
            "agentic_pipeline.adapters.processing_adapter.BookIngestionApp.create",
            return_value=mock_app,
        ):
            result = adapter.process_book(
                book_path="/fake/book.epub",
                save_to_storage=False,
            )

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.book_id == "test-123"
        assert result.quality_score == 85
        assert result.detection_confidence == 0.9

    def test_returns_failure_on_error(self, tmp_path):
        """process_book returns failure result when processing fails."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        with patch(
            "agentic_pipeline.adapters.processing_adapter.BookIngestionApp.create",
            side_effect=RuntimeError("Conversion failed"),
        ):
            result = adapter.process_book(
                book_path="/fake/book.epub",
                save_to_storage=False,
            )

        assert result.success is False
        assert "Conversion failed" in result.error

    def test_handles_unsuccessful_result(self, tmp_path):
        """process_book handles unsuccessful result from app."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.book_id = "test-456"
        mock_result.error = "Chapter detection failed"
        mock_result.pipeline_result = None

        mock_app = _make_mock_app(mock_result)

        with patch(
            "agentic_pipeline.adapters.processing_adapter.BookIngestionApp.create",
            return_value=mock_app,
        ):
            result = adapter.process_book(
                book_path="/fake/book.epub",
                save_to_storage=False,
            )

        assert result.success is False
        assert result.error == "Chapter detection failed"
        assert result.quality_score == 0


class TestProcessingAdapterFields:
    """Tests for ProcessingResult field calculations."""

    def test_word_count_sums_chapters(self, tmp_path):
        """Word count is sum of all chapter word counts."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        mock_result = _make_success_result(
            book_id="test-789",
            quality_score=75,
            chapters=[
                {"word_count": 1000},
                {"word_count": 2000},
                {"word_count": 3000},
            ],
        )
        mock_result.pipeline_result.detection_method = "pattern"
        mock_result.pipeline_result.needs_review = True
        mock_result.pipeline_result.warnings = ["low confidence"]

        mock_app = _make_mock_app(mock_result)

        with patch(
            "agentic_pipeline.adapters.processing_adapter.BookIngestionApp.create",
            return_value=mock_app,
        ):
            result = adapter.process_book(
                book_path="/fake/book.epub",
                save_to_storage=False,
            )

        assert result.word_count == 6000
        assert result.chapter_count == 3

    def test_handles_missing_word_count(self, tmp_path):
        """Word count handles chapters without word_count field."""
        db_path = tmp_path / "test.db"
        adapter = ProcessingAdapter(db_path=db_path)

        mock_result = _make_success_result(
            book_id="test-abc",
            quality_score=70,
            chapters=[
                {"title": "Chapter 1"},  # No word_count
                {"word_count": 2000},
            ],
        )
        mock_result.pipeline_result.detection_confidence = 0.75
        mock_result.pipeline_result.detection_method = "heuristic"
        mock_result.llm_fallback_used = True

        mock_app = _make_mock_app(mock_result)

        with patch(
            "agentic_pipeline.adapters.processing_adapter.BookIngestionApp.create",
            return_value=mock_app,
        ):
            result = adapter.process_book(
                book_path="/fake/book.epub",
                save_to_storage=False,
            )

        assert result.word_count == 2000
        assert result.llm_fallback_used is True


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_successful_result(self):
        """Successful embedding result has correct fields."""
        result = EmbeddingResult(
            success=True,
            chapters_processed=10,
        )
        assert result.success is True
        assert result.chapters_processed == 10
        assert result.error is None

    def test_failed_result(self):
        """Failed embedding result includes error."""
        result = EmbeddingResult(
            success=False,
            chapters_processed=0,
            error="torch not available",
        )
        assert result.success is False
        assert result.error == "torch not available"
