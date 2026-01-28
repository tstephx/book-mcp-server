"""Tests for approval actions."""

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


def test_approve_book(db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True

    pipeline = repo.get(pid)
    assert pipeline["state"] == PipelineState.APPROVED.value
    assert pipeline["approved_by"] == "human:taylor"


def test_reject_book(db_path):
    from agentic_pipeline.approval.actions import reject_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    result = reject_book(db_path, pid, reason="Poor quality extraction", actor="human:taylor")

    assert result["success"] is True

    pipeline = repo.get(pid)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_approve_creates_audit_record(db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    approve_book(db_path, pid, actor="human:taylor")

    # Check audit record
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM approval_audit WHERE pipeline_id = ?", (pid,))
    audit = cursor.fetchone()
    conn.close()

    assert audit is not None
    assert audit["action"] == "approved"
    assert audit["actor"] == "human:taylor"
