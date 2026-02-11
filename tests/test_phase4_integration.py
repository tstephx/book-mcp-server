# tests/test_phase4_integration.py
"""Integration tests for Phase 4 features."""

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


def test_full_batch_approve_flow(db_path):
    """Test complete batch approve workflow with audit."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter
    from agentic_pipeline.audit import AuditTrail

    repo = PipelineRepository(db_path)

    # Create books with different confidence
    for i in range(5):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        transition_to(repo, pid, PipelineState.PENDING_APPROVAL)
        repo.update_book_profile(pid, {
            "book_type": "technical_tutorial",
            "confidence": 0.85 + (i * 0.03)  # 0.85, 0.88, 0.91, 0.94, 0.97
        })

    # Batch approve high confidence
    ops = BatchOperations(db_path)
    filter = BatchFilter(min_confidence=0.9, book_type="technical_tutorial")

    # Preview first
    preview = ops.approve(filter, actor="human:test", execute=False)
    assert preview["would_approve"] == 3  # 0.91, 0.94, 0.97

    # Execute
    result = ops.approve(filter, actor="human:test", execute=True)
    assert result["approved"] == 3

    # Check audit
    trail = AuditTrail(db_path)
    entries = trail.query(action="BATCH_APPROVED")
    assert len(entries) == 1
    assert entries[0]["filter_used"]["min_confidence"] == 0.9


def test_health_with_stuck_detection(db_path):
    """Test health monitor integrates stuck detection."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor, StuckDetector
    import sqlite3
    from datetime import datetime, timezone, timedelta

    repo = PipelineRepository(db_path)

    # Create active pipeline
    pid = repo.create("/book.epub", "hash123")
    transition_to(repo, pid, PipelineState.PROCESSING)

    # Make it stuck
    conn = sqlite3.connect(db_path)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (old_time, pid))
    conn.commit()
    conn.close()

    # Check health includes stuck
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    stuck = detector.detect()

    assert report["active"] == 1
    assert len(stuck) == 1
    assert stuck[0]["id"] == pid


def test_priority_queue_ordering(db_path):
    """Test that priority queue orders correctly."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    id_low = repo.create("/low.epub", "hash1", priority=10)
    id_high = repo.create("/high.epub", "hash2", priority=1)
    id_med = repo.create("/med.epub", "hash3", priority=5)

    # Get queue
    queue = repo.find_by_state(PipelineState.DETECTED)

    # Should be ordered by priority
    assert queue[0]["id"] == id_high  # priority 1
    assert queue[1]["id"] == id_med   # priority 5
    assert queue[2]["id"] == id_low   # priority 10
