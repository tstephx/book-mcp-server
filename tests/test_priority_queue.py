# tests/test_priority_queue.py
"""Tests for priority queue functionality."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_create_with_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123", priority=2)

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 2


def test_default_priority_is_5(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 5


def test_find_by_state_orders_by_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    id_low = repo.create("/low.epub", "hash1", priority=10)
    id_high = repo.create("/high.epub", "hash2", priority=1)
    id_med = repo.create("/med.epub", "hash3", priority=5)

    results = repo.find_by_state(PipelineState.DETECTED)

    # Should be ordered: priority 1, then 5, then 10
    assert results[0]["id"] == id_high
    assert results[1]["id"] == id_med
    assert results[2]["id"] == id_low


def test_update_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123", priority=5)

    repo.update_priority(pid, 1)

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 1


def test_get_queue_by_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    repo.create("/a.epub", "hash1", priority=1)
    repo.create("/b.epub", "hash2", priority=1)
    repo.create("/c.epub", "hash3", priority=5)

    result = repo.get_queue_by_priority()

    assert result[1] == 2  # Two books at priority 1
    assert result[5] == 1  # One book at priority 5
