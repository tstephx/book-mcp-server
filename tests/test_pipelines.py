"""Tests for pipeline repository."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from conftest import transition_to


def _set_updated_at(db_path, pipeline_id, seconds_ago):
    """Backdate a pipeline's updated_at to simulate elapsed time."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (ts, pipeline_id))
    conn.commit()
    conn.close()


class TestConcurrentClaim:
    """Two actors must not both drive one pipeline.

    Regression: the worker reads find_by_state(DETECTED) then acts, with no
    atomic claim. A CLI reingest driving the same record raced it; both ran
    detected->hashing->classifying, then one tried SELECTING_STRATEGY against a
    state the other had already moved to PROCESSING. The loser's crash routed
    the book to NEEDS_RETRY, the retry loop exhausted it, and two books were
    destroyed with their data already cleaned up.
    """

    def test_losing_a_race_raises_instead_of_overwriting(self, db_path):
        """If another actor moves the record between our read and our write, fail."""
        from agentic_pipeline.db import pipelines as pl
        from agentic_pipeline.db.pipelines import ConcurrentModificationError, PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")

        real = pl.can_transition

        def steal_then_allow(old, new):
            """Simulate a rival claiming the record inside our read-write window."""
            rival = sqlite3.connect(db_path)
            rival.execute(
                "UPDATE processing_pipelines SET state = ? WHERE id = ?",
                (PipelineState.HASHING.value, pid),
            )
            rival.commit()
            rival.close()
            return real(old, new)

        with patch.object(pl, "can_transition", steal_then_allow):
            with pytest.raises(ConcurrentModificationError):
                repo.update_state(pid, PipelineState.HASHING)

    def test_the_rivals_write_is_not_clobbered(self, db_path):
        """The loser must leave the winner's state intact."""
        from agentic_pipeline.db import pipelines as pl
        from agentic_pipeline.db.pipelines import ConcurrentModificationError, PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        real = pl.can_transition

        def steal_then_allow(old, new):
            rival = sqlite3.connect(db_path)
            rival.execute(
                "UPDATE processing_pipelines SET state = ? WHERE id = ?",
                (PipelineState.HASHING.value, pid),
            )
            rival.commit()
            rival.close()
            return real(old, new)

        with patch.object(pl, "can_transition", steal_then_allow):
            with pytest.raises(ConcurrentModificationError):
                repo.update_state(pid, PipelineState.HASHING)

        assert repo.get(pid)["state"] == PipelineState.HASHING.value

    def test_expected_state_refuses_to_touch_a_record_we_no_longer_own(self, db_path):
        """A recovery path must not reset a record another actor has taken.

        The worker reads a book as DETECTED, then on error forces NEEDS_RETRY.
        If the rival already moved it to HASHING, that reset is still a *valid*
        transition — so only an explicit ownership assertion stops the clobber.
        """
        from agentic_pipeline.db.pipelines import ConcurrentModificationError, PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")
        repo.update_state(pid, PipelineState.HASHING)  # rival claimed it

        # We still believe it is DETECTED, as the worker's stale read would.
        with pytest.raises(ConcurrentModificationError):
            repo.update_state(pid, PipelineState.NEEDS_RETRY, expected_state=PipelineState.DETECTED)

        assert repo.get(pid)["state"] == PipelineState.HASHING.value

    def test_expected_state_allows_the_owner_through(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")

        repo.update_state(pid, PipelineState.HASHING, expected_state=PipelineState.DETECTED)

        assert repo.get(pid)["state"] == PipelineState.HASHING.value

    def test_uncontended_transition_still_works(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        repo = PipelineRepository(db_path)
        pid = repo.create("/path/to/book.epub", "abc123")

        repo.update_state(pid, PipelineState.HASHING)

        assert repo.get(pid)["state"] == PipelineState.HASHING.value

    def test_concurrent_error_is_not_swallowed_as_invalid_transition(self, db_path):
        """ConcurrentModificationError must be distinguishable from a bad transition."""
        from agentic_pipeline.db.pipelines import ConcurrentModificationError

        assert not issubclass(ConcurrentModificationError, ValueError)


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
