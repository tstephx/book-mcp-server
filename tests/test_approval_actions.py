"""Tests for approval actions."""

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


def _mock_embedding_result(success=True, chapters_processed=5, error=None):
    """Create a mock EmbeddingResult."""
    from agentic_pipeline.adapters.processing_adapter import EmbeddingResult
    return EmbeddingResult(
        success=success,
        chapters_processed=chapters_processed,
        error=error,
    )


def _create_pending_pipeline(db_path, processing_result=None):
    """Create a pipeline in PENDING_APPROVAL state via valid transitions."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    # Walk through valid transitions to reach PENDING_APPROVAL
    for state in [
        PipelineState.HASHING,
        PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY,
        PipelineState.PROCESSING,
        PipelineState.VALIDATING,
        PipelineState.PENDING_APPROVAL,
    ]:
        repo.update_state(pid, state)
    if processing_result is not None:
        repo.update_processing_result(pid, processing_result)
    return pid


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_book(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid = _create_pending_pipeline(db_path)
    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    assert result["state"] == PipelineState.COMPLETE.value
    assert result["chapters_embedded"] == 5

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.COMPLETE.value
    assert pipeline["approved_by"] == "human:taylor"


def test_reject_book(db_path):
    from agentic_pipeline.approval.actions import reject_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    pid = _create_pending_pipeline(db_path)
    result = reject_book(db_path, pid, reason="Poor quality extraction", actor="human:taylor")

    assert result["success"] is True

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.REJECTED.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_creates_audit_record(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.pipeline.states import PipelineState
    import sqlite3

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid = _create_pending_pipeline(db_path)
    approve_book(db_path, pid, actor="human:taylor")

    # Check audit record
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM approval_audit WHERE pipeline_id = ?", (pid,))
    audit = cursor.fetchone()
    conn.close()

    assert audit is not None
    assert audit["action"] == "approved"
    assert audit["actor"] == "human:taylor"


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_embedding_failure_goes_to_needs_retry(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result(
        success=False, chapters_processed=0, error="OpenAI rate limit"
    )

    pid = _create_pending_pipeline(db_path)
    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert "OpenAI rate limit" in result["embedding_error"]

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.NEEDS_RETRY.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_embedding_exception_goes_to_needs_retry(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.side_effect = RuntimeError("Connection refused")

    pid = _create_pending_pipeline(db_path)
    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert "Connection refused" in result["embedding_error"]

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.NEEDS_RETRY.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_no_processing_result_uses_pipeline_id(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    # No processing_result set — should fall back to pipeline_id as book_id
    pid = _create_pending_pipeline(db_path, processing_result=None)
    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    assert result["state"] == PipelineState.COMPLETE.value
    mock_instance.generate_embeddings.assert_called_once_with(book_id=pid)


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_approve_uses_book_id_from_processing_result(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid = _create_pending_pipeline(
        db_path, processing_result={"book_id": "my-book-uuid"}
    )
    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    mock_instance.generate_embeddings.assert_called_once_with(book_id="my-book-uuid")
