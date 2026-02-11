# tests/test_batch_filters.py
"""Tests for batch filter functionality."""

import pytest
import tempfile
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


def test_filter_by_min_confidence(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    # Create pipelines with different confidence
    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.95})

    id2 = repo.create("/book2.epub", "hash2")
    transition_to(repo, id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "technical_tutorial", "confidence": 0.7})

    filter = BatchFilter(min_confidence=0.9)
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id1


def test_filter_by_book_type(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.9})

    id2 = repo.create("/book2.epub", "hash2")
    transition_to(repo, id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "newspaper", "confidence": 0.9})

    filter = BatchFilter(book_type="technical_tutorial")
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id1


def test_filter_by_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)

    id2 = repo.create("/book2.epub", "hash2")
    transition_to(repo, id2, PipelineState.NEEDS_RETRY)

    filter = BatchFilter(state="needs_retry")
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id2


def test_filter_max_count(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    for i in range(10):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        transition_to(repo, pid, PipelineState.PENDING_APPROVAL)

    filter = BatchFilter(max_count=3)
    results = filter.apply(db_path)

    assert len(results) == 3


def test_filter_to_dict(db_path):
    from agentic_pipeline.batch.filters import BatchFilter

    filter = BatchFilter(min_confidence=0.9, book_type="technical_tutorial")
    d = filter.to_dict()

    assert d["min_confidence"] == 0.9
    assert d["book_type"] == "technical_tutorial"
