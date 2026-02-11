# tests/test_pipelines_extended.py
"""Extended tests for pipeline repository."""

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


def test_find_by_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create pipelines in different states
    id1 = repo.create("/book1.epub", "hash1")
    id2 = repo.create("/book2.epub", "hash2")
    transition_to(repo, id2, PipelineState.CLASSIFYING)

    detected = repo.find_by_state(PipelineState.DETECTED)

    assert len(detected) == 1
    assert detected[0]["id"] == id1


def test_increment_retry_count(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")

    count1 = repo.increment_retry_count(pid)
    count2 = repo.increment_retry_count(pid)

    assert count1 == 1
    assert count2 == 2


def test_find_by_state_with_limit(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create 5 pipelines
    for i in range(5):
        repo.create(f"/book{i}.epub", f"hash{i}")

    # Get only 2
    results = repo.find_by_state(PipelineState.DETECTED, limit=2)

    assert len(results) == 2
