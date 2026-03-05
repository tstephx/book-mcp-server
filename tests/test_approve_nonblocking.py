"""Tests that approve_book() returns promptly without blocking on embedding."""

import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def _create_pending_pipeline(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash_nonblocking")
    for state in [
        PipelineState.HASHING,
        PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY,
        PipelineState.PROCESSING,
        PipelineState.VALIDATING,
        PipelineState.PENDING_APPROVAL,
    ]:
        repo.update_state(pid, state)
    return pid


def test_approve_book_returns_before_embedding_completes(db_path):
    """approve_book() must return in under 1s even if embedding takes a long time.

    Uses a threading.Event so the background thread exits cleanly before DB
    teardown — avoids PytestUnhandledThreadExceptionWarning in later tests.
    """
    from agentic_pipeline.approval.actions import approve_book

    # Event that the background thread blocks on; the test signals it to finish.
    release = threading.Event()

    def slow_embed(*args, **kwargs):
        release.wait(timeout=10)  # blocks until test signals, or 10s safety timeout

    with patch(
        "agentic_pipeline.approval.actions._run_embedding_background",
        side_effect=slow_embed,
    ):
        pid = _create_pending_pipeline(db_path)
        start = time.monotonic()
        result = approve_book(db_path, pid, actor="human:taylor")
        elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"approve_book blocked for {elapsed:.2f}s — embedding is still synchronous"
    assert result["success"] is True
    assert result["state"] == "approved"

    # Unblock the daemon thread so it exits before pytest tears down the DB
    release.set()
    time.sleep(0.05)  # give the thread a moment to observe the signal and exit


def test_approve_book_transitions_to_approved_state(db_path):
    """Pipeline must reach APPROVED state immediately after approve_book() returns."""
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    with patch("agentic_pipeline.approval.actions._run_embedding_background"):
        pid = _create_pending_pipeline(db_path)
        approve_book(db_path, pid, actor="human:taylor")

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] in (
        PipelineState.APPROVED.value,
        PipelineState.EMBEDDING.value,
        PipelineState.COMPLETE.value,
    ), f"Unexpected state: {pipeline['state']}"
