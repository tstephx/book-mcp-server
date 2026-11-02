"""EMBEDDING_TIMEOUT_SECONDS must be enforced on the live EMBEDDING path (#32).

The live path is approval.actions._complete_approved, reached from
approve_book() (background thread or background=False) and from the
orchestrator's retry/worker loop.
"""

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from agentic_pipeline.pipeline.states import PipelineState

from tests.test_approval_actions import _create_pending_pipeline, _setup_approved_pipeline


@pytest.fixture
def db_path(tmp_path):
    from agentic_pipeline.db.migrations import run_migrations

    path = tmp_path / "pipeline.db"
    run_migrations(path)
    return path


@pytest.fixture
def MockAdapter():
    """Stand-in processing_adapter module, so tests run without book_ingestion installed."""
    fake_mod = types.ModuleType("agentic_pipeline.adapters.processing_adapter")
    fake_mod.ProcessingAdapter = MagicMock(name="ProcessingAdapter")
    with patch.dict(sys.modules, {"agentic_pipeline.adapters.processing_adapter": fake_mod}):
        yield fake_mod.ProcessingAdapter


def _history_error(db_path, pid):
    from agentic_pipeline.db.connection import get_pipeline_db

    with get_pipeline_db(db_path) as conn:
        row = conn.execute(
            "SELECT error_details FROM pipeline_state_history "
            "WHERE pipeline_id = ? AND to_state = 'needs_retry' ORDER BY id DESC LIMIT 1",
            (pid,),
        ).fetchone()
    import json

    return json.loads(row["error_details"])


def _ok_result():
    return types.SimpleNamespace(success=True, chapters_processed=5, error=None)


def _hang(release):
    def slow(*args, **kwargs):
        release.wait(timeout=3)
        # Eventually "succeeds": without a timeout the book would reach COMPLETE,
        # so the NEEDS_RETRY assertions can't pass for an unrelated reason.
        return _ok_result()

    return slow


def test_hung_embedding_times_out_to_needs_retry(MockAdapter, db_path, monkeypatch):
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository

    release = threading.Event()
    MockAdapter.return_value.generate_embeddings.side_effect = _hang(release)
    pid, pipeline = _setup_approved_pipeline(db_path)

    start = time.monotonic()
    try:
        result = _complete_approved(db_path, pid, pipeline, embedding_timeout=1)
    finally:
        release.set()

    assert time.monotonic() - start < 2.5, "timeout was not enforced; call hung"
    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert "timeout" in result["embedding_error"].lower()
    assert PipelineRepository(db_path).get(pid)["state"] == PipelineState.NEEDS_RETRY.value
    assert _history_error(db_path, pid)["failed_in"] == "embedding"


def test_timeout_defaults_to_env_config(MockAdapter, db_path, monkeypatch):
    """With no explicit value, EMBEDDING_TIMEOUT_SECONDS drives the timeout."""
    from agentic_pipeline.approval.actions import _complete_approved

    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "1")
    release = threading.Event()
    MockAdapter.return_value.generate_embeddings.side_effect = _hang(release)
    pid, pipeline = _setup_approved_pipeline(db_path)

    try:
        result = _complete_approved(db_path, pid, pipeline)
    finally:
        release.set()

    assert result["state"] == PipelineState.NEEDS_RETRY.value


def test_approve_book_synchronous_entry_point_times_out(MockAdapter, db_path, monkeypatch):
    """approve_book(background=False) (the CLI path) reaches the same timeout."""
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository

    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "1")
    release = threading.Event()
    MockAdapter.return_value.generate_embeddings.side_effect = _hang(release)
    pid = _create_pending_pipeline(db_path)
    try:
        result = approve_book(db_path, pid, actor="human:taylor", background=False)
    finally:
        release.set()

    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert PipelineRepository(db_path).get(pid)["state"] == PipelineState.NEEDS_RETRY.value


def test_orchestrator_passes_its_configured_timeout(MockAdapter, db_path):
    """Orchestrator._complete_approved honours config.embedding_timeout."""
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator

    release = threading.Event()
    MockAdapter.return_value.generate_embeddings.side_effect = _hang(release)
    pid, _ = _setup_approved_pipeline(db_path)

    orch = Orchestrator(OrchestratorConfig(db_path=db_path, embedding_timeout=1))
    try:
        result = orch._complete_approved(pid)
    finally:
        release.set()

    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert result["pipeline_id"] == pid


def test_fast_embedding_unaffected_by_timeout(MockAdapter, db_path):
    from agentic_pipeline.approval.actions import _complete_approved
    MockAdapter.return_value.generate_embeddings.return_value = _ok_result()
    pid, pipeline = _setup_approved_pipeline(db_path)

    result = _complete_approved(db_path, pid, pipeline, embedding_timeout=5)

    assert result["state"] == PipelineState.COMPLETE.value
