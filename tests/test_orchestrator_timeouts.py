"""Tests for processing and embedding timeout enforcement in the orchestrator."""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def orchestrator(db_path):
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig(db_path=db_path, processing_timeout=1, embedding_timeout=1)
    return Orchestrator(config)


def test_processing_timeout_raises_pipeline_timeout_error(orchestrator):
    """_run_processing must raise PipelineTimeoutError when adapter hangs past processing_timeout."""
    from agentic_pipeline.orchestrator.errors import PipelineTimeoutError

    def slow_process(*args, **kwargs):
        time.sleep(5)  # Far exceeds timeout=1

    with patch.object(orchestrator.processing_adapter, "process_book", side_effect=slow_process):
        with pytest.raises(PipelineTimeoutError):
            orchestrator._run_processing("/fake/book.epub", book_id="test-id")


def test_embedding_timeout_raises_pipeline_timeout_error(orchestrator):
    """_run_embedding must raise PipelineTimeoutError when adapter hangs past embedding_timeout."""
    from agentic_pipeline.orchestrator.errors import PipelineTimeoutError

    def slow_embed(*args, **kwargs):
        time.sleep(5)

    with patch.object(orchestrator.processing_adapter, "generate_embeddings", side_effect=slow_embed):
        with pytest.raises(PipelineTimeoutError):
            orchestrator._run_embedding(book_id="test-id")


def test_processing_completes_within_timeout(orchestrator):
    """_run_processing succeeds when adapter responds before timeout."""
    from agentic_pipeline.adapters.processing_adapter import ProcessingResult

    fast_result = ProcessingResult(
        success=True,
        book_id="test-id",
        quality_score=0.9,
        detection_confidence=0.85,
        detection_method="direct",
        needs_review=False,
        warnings=[],
        chapter_count=10,
        word_count=50000,
        llm_fallback_used=False,
    )

    with patch.object(orchestrator.processing_adapter, "process_book", return_value=fast_result):
        result = orchestrator._run_processing("/fake/book.epub", book_id="test-id")
        assert result["book_id"] == "test-id"
        assert result["quality_score"] == 0.9
