# tests/test_batch_operations.py
"""Tests for batch operations."""

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


def test_batch_approve(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter
    from agentic_pipeline.audit import AuditTrail

    repo = PipelineRepository(db_path)

    # Create pending approval books
    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.95})

    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "technical_tutorial", "confidence": 0.92})

    ops = BatchOperations(db_path)
    filter = BatchFilter(min_confidence=0.9)

    result = ops.approve(filter, actor="human:taylor", execute=True)

    assert result["approved"] == 2

    # Check audit trail
    trail = AuditTrail(db_path)
    entries = trail.query(action="BATCH_APPROVED")
    assert len(entries) == 1
    assert entries[0]["filter_used"]["min_confidence"] == 0.9


def test_batch_approve_dry_run(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)

    ops = BatchOperations(db_path)
    filter = BatchFilter()

    result = ops.approve(filter, actor="human:taylor", execute=False)

    assert result["would_approve"] == 1
    assert result["approved"] == 0

    # State should not change
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.PENDING_APPROVAL.value


def test_batch_reject(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "newspaper", "confidence": 0.9})

    ops = BatchOperations(db_path)
    filter = BatchFilter(book_type="newspaper")

    result = ops.reject(filter, reason="Not ingesting periodicals", actor="human:taylor", execute=True)

    assert result["rejected"] == 1

    # Check state changed
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_batch_set_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1", priority=5)

    ops = BatchOperations(db_path)
    filter = BatchFilter(state="detected")

    result = ops.set_priority(filter, priority=1, actor="human:taylor", execute=True)

    assert result["updated"] == 1

    pipeline = repo.get(id1)
    assert pipeline["priority"] == 1
