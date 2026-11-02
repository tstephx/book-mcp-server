"""Books stranded in EMBEDDING by a worker crash are recovered at startup (#34).

approve_book() runs EMBEDDING on a daemon thread, so a killed worker leaves the
book in EMBEDDING forever. Worker startup must move such books to NEEDS_RETRY
(failed_in=embedding) so the existing retry loop finishes them.
"""

import json
import sqlite3
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agentic_pipeline.pipeline.states import PipelineState
from tests.test_approval_actions import _setup_approved_pipeline

CUTOFF = 600


@pytest.fixture
def db_path(tmp_path):
    from agentic_pipeline.db.migrations import run_migrations

    path = tmp_path / "pipeline.db"
    run_migrations(path)
    return path


@pytest.fixture
def MockAdapter():
    fake_mod = types.ModuleType("agentic_pipeline.adapters.processing_adapter")
    fake_mod.ProcessingAdapter = MagicMock(name="ProcessingAdapter")
    with patch.dict(sys.modules, {"agentic_pipeline.adapters.processing_adapter": fake_mod}):
        yield fake_mod.ProcessingAdapter


def _backdate(db_path, pid, seconds_ago):
    ts = None if seconds_ago is None else (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (ts, pid))
    conn.commit()
    conn.close()


def _embedding_book(db_path, seconds_ago, via_approval=False):
    """A book sitting in EMBEDDING, last touched seconds_ago (None -> NULL).

    via_approval=True builds it through the real approve flow (needed when the
    book must be completable); otherwise a unique-hash row is walked there.
    """
    import uuid

    from agentic_pipeline.db.pipelines import PipelineRepository
    from conftest import transition_to

    repo = PipelineRepository(db_path)
    if via_approval:
        pid, _ = _setup_approved_pipeline(db_path)
        repo.update_state(pid, PipelineState.EMBEDDING)
    else:
        pid = repo.create(f"/{uuid.uuid4().hex}.epub", uuid.uuid4().hex)
        transition_to(repo, pid, PipelineState.EMBEDDING)
    _backdate(db_path, pid, seconds_ago)
    return repo, pid


def _recovery_error(db_path, pid):
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT from_state, error_details FROM pipeline_state_history "
        "WHERE pipeline_id = ? AND to_state = 'needs_retry' ORDER BY rowid DESC LIMIT 1",
        (pid,),
    ).fetchone()
    conn.close()
    return row[0], json.loads(row[1])


def test_stale_embedding_book_moves_to_needs_retry(db_path):
    repo, pid = _embedding_book(db_path, seconds_ago=CUTOFF + 60)

    count = repo.reset_stale_embedding(CUTOFF)

    assert count == 1 and isinstance(count, int)
    assert repo.get(pid)["state"] == PipelineState.NEEDS_RETRY.value
    from_state, err = _recovery_error(db_path, pid)
    assert from_state == PipelineState.EMBEDDING.value
    assert err["failed_in"] == "embedding"
    assert isinstance(err["embedding_error"], str) and "crashed" in err["embedding_error"]


def test_fresh_embedding_book_untouched(db_path):
    repo, pid = _embedding_book(db_path, seconds_ago=5)

    assert repo.reset_stale_embedding(CUTOFF) == 0
    assert repo.get(pid)["state"] == PipelineState.EMBEDDING.value


def test_threshold_boundary(db_path):
    repo, just_under = _embedding_book(db_path, seconds_ago=CUTOFF - 5)
    _, just_over = _embedding_book(db_path, seconds_ago=CUTOFF + 5)

    assert repo.reset_stale_embedding(CUTOFF) == 1
    assert repo.get(just_under)["state"] == PipelineState.EMBEDDING.value
    assert repo.get(just_over)["state"] == PipelineState.NEEDS_RETRY.value


def test_null_updated_at_untouched(db_path):
    repo, pid = _embedding_book(db_path, seconds_ago=None)

    assert repo.reset_stale_embedding(CUTOFF) == 0
    assert repo.get(pid)["state"] == PipelineState.EMBEDDING.value


@pytest.mark.parametrize("state", [s for s in PipelineState if s is not PipelineState.EMBEDDING])
def test_other_states_untouched(db_path, state):
    """Only EMBEDDING is swept — every other state is left alone, however old."""
    repo, pid = _embedding_book(db_path, seconds_ago=3600)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE processing_pipelines SET state = ? WHERE id = ?", (state.value, pid))
    conn.commit()
    conn.close()
    _backdate(db_path, pid, 3600)

    assert repo.reset_stale_embedding(CUTOFF) == 0
    assert repo.get(pid)["state"] == state.value


def test_row_moved_by_another_actor_is_skipped_not_raised(db_path):
    repo, rival_wins = _embedding_book(db_path, seconds_ago=CUTOFF + 60)
    _, ours = _embedding_book(db_path, seconds_ago=CUTOFF + 60)
    real = repo.update_state
    stolen = []

    def update_state(pid, new_state, *args, **kwargs):
        if not stolen:  # a live embed finishes just before our first write
            stolen.append(pid)
            real(pid, PipelineState.COMPLETE)
        return real(pid, new_state, *args, **kwargs)

    repo.update_state = update_state

    count = repo.reset_stale_embedding(CUTOFF)

    assert count == 1, "only the row we actually moved is counted"
    assert repo.get(stolen[0])["state"] == PipelineState.COMPLETE.value, "rival's write not clobbered"
    other = ({rival_wins, ours} - set(stolen)).pop()
    assert repo.get(other)["state"] == PipelineState.NEEDS_RETRY.value


def test_retry_one_finishes_recovered_book(MockAdapter, db_path):
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator

    MockAdapter.return_value.generate_embeddings.return_value = types.SimpleNamespace(
        success=True, chapters_processed=5, error=None
    )
    repo, pid = _embedding_book(db_path, seconds_ago=CUTOFF + 60, via_approval=True)
    repo.reset_stale_embedding(CUTOFF)

    Orchestrator(OrchestratorConfig(db_path=db_path))._retry_one(repo.get(pid))

    assert repo.get(pid)["state"] == PipelineState.COMPLETE.value


def test_run_worker_startup_recovers_stranded_book(MockAdapter, db_path):
    """Wired into run_worker with cutoff = embedding_timeout + margin."""
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.db.pipelines import EMBEDDING_STALE_MARGIN_SECONDS

    cfg = OrchestratorConfig(db_path=db_path, embedding_timeout=100)
    window = 100 + EMBEDDING_STALE_MARGIN_SECONDS
    repo, stale = _embedding_book(db_path, seconds_ago=window + 30)
    _, live = _embedding_book(db_path, seconds_ago=window - 30)

    from agentic_pipeline.orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(cfg)
    orch.shutdown_requested = True  # skip the poll loop; startup recovery only
    orch.run_worker()

    assert repo.get(stale)["state"] == PipelineState.NEEDS_RETRY.value
    assert repo.get(live)["state"] == PipelineState.EMBEDDING.value
