# tests/test_audit_trail.py
"""Tests for audit trail functionality."""

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


def test_log_approval(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="book123",
        pipeline_id="pipe456",
        action="APPROVED",
        actor="human:taylor",
        confidence=0.95
    )

    entries = trail.query(book_id="book123")
    assert len(entries) == 1
    assert entries[0]["action"] == "APPROVED"
    assert entries[0]["actor"] == "human:taylor"


def test_log_rejection_with_reason(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="book123",
        action="REJECTED",
        actor="human:taylor",
        reason="Not a technical book"
    )

    entries = trail.query(book_id="book123")
    assert entries[0]["reason"] == "Not a technical book"


def test_query_by_actor(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(book_id="a", action="APPROVED", actor="human:taylor")
    trail.log(book_id="b", action="APPROVED", actor="auto:high_confidence")
    trail.log(book_id="c", action="APPROVED", actor="human:taylor")

    entries = trail.query(actor="human:taylor")
    assert len(entries) == 2


def test_query_by_action(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(book_id="a", action="APPROVED", actor="human:taylor")
    trail.log(book_id="b", action="REJECTED", actor="human:taylor")

    entries = trail.query(action="REJECTED")
    assert len(entries) == 1
    assert entries[0]["book_id"] == "b"


def test_query_with_limit(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    for i in range(10):
        trail.log(book_id=f"book{i}", action="APPROVED", actor="auto:test")

    entries = trail.query(limit=5)
    assert len(entries) == 5


def test_log_batch_operation_with_filter(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="batch",
        action="BATCH_APPROVED",
        actor="batch:filter_abc",
        filter_used={"min_confidence": 0.9, "book_type": "technical_tutorial"}
    )

    entries = trail.query(action="BATCH_APPROVED")
    assert entries[0]["filter_used"]["min_confidence"] == 0.9
