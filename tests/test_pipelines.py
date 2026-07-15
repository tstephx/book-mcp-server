"""Tests for pipeline repository."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

from conftest import transition_to


def _set_updated_at(db_path, pipeline_id, seconds_ago):
    """Backdate a pipeline's updated_at to simulate elapsed time."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (ts, pipeline_id))
    conn.commit()
    conn.close()


class TestResetStaleProcessing:
    """Recovering crashed books must not steal books another actor is processing.

    Regression: a worker's startup reset_stale_processing() force-reset an
    in-flight CLI reingest from processing -> detected, both actors then drove
    the same record, and the collision destroyed two books. The guard was
    `last_heartbeat IS NULL`, but nothing ever writes last_heartbeat, so it
    matched every row.
    """

    def test_resets_a_book_abandoned_long_ago(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        transition_to(repo, pid, PipelineState.PROCESSING)
        _set_updated_at(db_path, pid, seconds_ago=3600)

        count = repo.reset_stale_processing()

        assert count == 1
        assert repo.get(pid)["state"] == PipelineState.DETECTED.value

    def test_leaves_an_in_flight_book_alone(self, db_path):
        """The book another process is actively working must not be reset."""
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        transition_to(repo, pid, PipelineState.PROCESSING)  # updated_at = now

        count = repo.reset_stale_processing()

        assert count == 0
        assert repo.get(pid)["state"] == PipelineState.PROCESSING.value

    def test_does_not_reset_at_exactly_the_threshold(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.db.pipelines import STALE_PROCESSING_SECONDS
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        transition_to(repo, pid, PipelineState.PROCESSING)
        _set_updated_at(db_path, pid, seconds_ago=STALE_PROCESSING_SECONDS - 5)

        assert repo.reset_stale_processing() == 0
        assert repo.get(pid)["state"] == PipelineState.PROCESSING.value

    def test_resets_just_past_the_threshold(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.db.pipelines import STALE_PROCESSING_SECONDS
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        transition_to(repo, pid, PipelineState.PROCESSING)
        _set_updated_at(db_path, pid, seconds_ago=STALE_PROCESSING_SECONDS + 5)

        assert repo.reset_stale_processing() == 1
        assert repo.get(pid)["state"] == PipelineState.DETECTED.value

    def test_ignores_books_in_other_states(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        transition_to(repo, pid, PipelineState.VALIDATING)
        _set_updated_at(db_path, pid, seconds_ago=3600)

        assert repo.reset_stale_processing() == 0
        assert repo.get(pid)["state"] == PipelineState.VALIDATING.value


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

    pipeline_id = repo.create(source_path="/path/to/book.epub", content_hash="abc123")

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

    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)
    transition_to(repo, id2, PipelineState.PENDING_APPROVAL)
    transition_to(repo, id3, PipelineState.COMPLETE)

    pending = repo.list_pending_approval()
    assert len(pending) == 2
    assert all(p["state"] == PipelineState.PENDING_APPROVAL.value for p in pending)
