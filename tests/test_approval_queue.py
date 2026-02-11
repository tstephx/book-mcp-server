"""Tests for approval queue."""

import pytest
import tempfile
import json
from pathlib import Path

from conftest import transition_to


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_get_pending_returns_formatted_queue(db_path):
    from agentic_pipeline.approval.queue import ApprovalQueue
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Setup: create a pending pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/path/to/book.epub", "hash123")
    repo.update_book_profile(pid, {
        "book_type": "technical_tutorial",
        "confidence": 0.92,
        "suggested_tags": ["ai", "python"]
    })
    transition_to(repo, pid, PipelineState.PENDING_APPROVAL)

    # Test
    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    assert result["pending_count"] == 1
    assert len(result["books"]) == 1
    assert result["books"][0]["source_path"] == "/path/to/book.epub"
    assert result["books"][0]["book_type"] == "technical_tutorial"


def test_get_pending_calculates_stats(db_path):
    from agentic_pipeline.approval.queue import ApprovalQueue
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create multiple pipelines
    for i, conf in enumerate([0.95, 0.85, 0.72]):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_book_profile(pid, {"confidence": conf, "book_type": "technical_tutorial"})
        transition_to(repo, pid, PipelineState.PENDING_APPROVAL)

    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    assert result["pending_count"] == 3
    assert result["stats"]["high_confidence"] == 1  # >= 0.9
    assert result["stats"]["needs_attention"] == 1  # < 0.8
