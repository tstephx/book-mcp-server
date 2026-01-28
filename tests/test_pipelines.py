"""Tests for pipeline repository."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime


@pytest.fixture
def db_path():
    """Create a temporary database."""
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_create_pipeline(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    pipeline_id = repo.create(
        source_path="/path/to/book.epub",
        content_hash="abc123"
    )

    assert pipeline_id is not None

    pipeline = repo.get(pipeline_id)
    assert pipeline["source_path"] == "/path/to/book.epub"
    assert pipeline["content_hash"] == "abc123"
    assert pipeline["state"] == PipelineState.DETECTED.value


def test_update_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pipeline_id = repo.create("/path/to/book.epub", "abc123")

    repo.update_state(pipeline_id, PipelineState.HASHING)

    pipeline = repo.get(pipeline_id)
    assert pipeline["state"] == PipelineState.HASHING.value


def test_find_by_hash(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pipeline_id = repo.create("/path/to/book.epub", "unique_hash_123")

    found = repo.find_by_hash("unique_hash_123")
    assert found is not None
    assert found["id"] == pipeline_id

    not_found = repo.find_by_hash("nonexistent")
    assert not_found is None


def test_list_pending_approval(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create pipelines in various states
    id1 = repo.create("/book1.epub", "hash1")
    id2 = repo.create("/book2.epub", "hash2")
    id3 = repo.create("/book3.epub", "hash3")

    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_state(id3, PipelineState.COMPLETE)

    pending = repo.list_pending_approval()
    assert len(pending) == 2
    assert all(p["state"] == PipelineState.PENDING_APPROVAL.value for p in pending)
