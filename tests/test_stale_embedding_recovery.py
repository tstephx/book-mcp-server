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


# --- Periodic sweep: a crashed MCP/CLI process strands books too (#34 review) ---


def test_sweep_is_throttled_between_calls(MockAdapter, db_path, monkeypatch):
    """_sweep_stranded_embedding_if_due runs the sweep at most once per interval."""
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator import orchestrator as orch_mod
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator

    clock = [1000.0]
    monkeypatch.setattr(orch_mod.time, "monotonic", lambda: clock[0])
    orch = Orchestrator(OrchestratorConfig(db_path=db_path, embedding_timeout=100))
    window = 100 + orch_mod.EMBEDDING_STALE_MARGIN_SECONDS

    assert orch._sweep_stranded_embedding_if_due() == 0  # first call is due; nothing stranded
    repo, pid = _embedding_book(db_path, seconds_ago=window + 30)

    clock[0] += orch_mod.EMBEDDING_SWEEP_INTERVAL_SECONDS - 1
    assert orch._sweep_stranded_embedding_if_due() == 0, "not due yet"
    assert repo.get(pid)["state"] == PipelineState.EMBEDDING.value

    clock[0] += 1
    assert orch._sweep_stranded_embedding_if_due() == 1, "due at the interval"
    assert repo.get(pid)["state"] == PipelineState.NEEDS_RETRY.value


def test_run_worker_loop_recovers_book_stranded_after_startup(MockAdapter, db_path, monkeypatch):
    """A book stranded mid-run (e.g. the MCP server crashed) is swept without a worker restart."""
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator import orchestrator as orch_mod
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator

    monkeypatch.setattr(orch_mod, "EMBEDDING_SWEEP_INTERVAL_SECONDS", 0)
    orch = Orchestrator(OrchestratorConfig(db_path=db_path, embedding_timeout=100))
    window = 100 + orch_mod.EMBEDDING_STALE_MARGIN_SECONDS
    stranded = {}

    def fake_sleep(_):
        # Runs on the first idle poll, i.e. after startup recovery has finished.
        repo, stranded["pid"] = _embedding_book(db_path, seconds_ago=window + 30)
        stranded["repo"] = repo

    def stop_after_retry(book):
        orch.shutdown_requested = True
        return {}

    monkeypatch.setattr(orch_mod.time, "sleep", fake_sleep)
    orch._retry_one = stop_after_retry
    orch.run_worker()

    assert stranded["repo"].get(stranded["pid"])["state"] == PipelineState.NEEDS_RETRY.value
