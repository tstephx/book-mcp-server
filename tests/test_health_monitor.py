# tests/test_health_monitor.py
"""Tests for health monitoring."""

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


def test_health_report_empty_queue(db_path):
    from agentic_pipeline.health import HealthMonitor

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["active"] == 0
    assert report["queued"] == 0
    assert report["stuck"] == []
    assert report["status"] == "idle"


def test_health_report_with_queued(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    repo.create("/book1.epub", "hash1", priority=1)
    repo.create("/book2.epub", "hash2", priority=5)
    repo.create("/book3.epub", "hash3", priority=5)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["queued"] == 3
    assert report["queue_by_priority"] == {1: 1, 5: 2}


def test_health_report_with_active(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    pid = repo.create("/book1.epub", "hash1")
    transition_to(repo, pid, PipelineState.PROCESSING)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["active"] == 1
    assert report["status"] == "processing"


def test_health_report_with_failed(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    pid = repo.create("/book1.epub", "hash1")
    transition_to(repo, pid, PipelineState.NEEDS_RETRY)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["failed"] == 1


def test_alerts_queue_backup(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    for i in range(150):
        repo.create(f"/book{i}.epub", f"hash{i}")

    monitor = HealthMonitor(db_path, alert_queue_threshold=100)
    report = monitor.get_health()

    assert any(a["type"] == "queue_backup" for a in report["alerts"])


def test_health_report_has_permanently_failed_count(db_path):
    """get_health() returns a 'permanently_failed' count distinct from 'failed'."""
    from agentic_pipeline.health import HealthMonitor
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash-abc")
    repo.update_state(pid, PipelineState.NEEDS_RETRY)
    repo.update_state(pid, PipelineState.FAILED)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert "permanently_failed" in report
    assert report["permanently_failed"] == 1


def test_status_is_not_idle_when_permanently_failed(db_path):
    """status must not be 'idle' when permanently_failed books exist."""
    from agentic_pipeline.health import HealthMonitor
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash-xyz")
    repo.update_state(pid, PipelineState.NEEDS_RETRY)
    repo.update_state(pid, PipelineState.FAILED)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["status"] != "idle"
    assert report["permanently_failed"] == 1


def test_permanently_failed_counted_in_failure_rate_alert(db_path):
    """permanently_failed books should count toward failure rate alerts."""
    from agentic_pipeline.health import HealthMonitor
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    # Create 15 completed + 4 permanently_failed → rate = 4/19 = 21% > 20% threshold
    for i in range(15):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_state(pid, PipelineState.HASHING)
        repo.update_state(pid, PipelineState.CLASSIFYING)
        repo.update_state(pid, PipelineState.SELECTING_STRATEGY)
        repo.update_state(pid, PipelineState.PROCESSING)
        repo.update_state(pid, PipelineState.VALIDATING)
        repo.update_state(pid, PipelineState.PENDING_APPROVAL)
        repo.update_state(pid, PipelineState.APPROVED)
        repo.update_state(pid, PipelineState.EMBEDDING)
        repo.update_state(pid, PipelineState.COMPLETE)
    for i in range(4):
        pid = repo.create(f"/fail{i}.epub", f"failhash{i}")
        repo.update_state(pid, PipelineState.NEEDS_RETRY)
        repo.update_state(pid, PipelineState.FAILED)

    monitor = HealthMonitor(db_path, alert_failure_rate=0.20)
    report = monitor.get_health()

    assert any(a["type"] == "high_failure_rate" for a in report["alerts"])


def test_failure_rate_uses_consistent_time_window(db_path):
    """All-time permanently_failed should not distort 24h failure rate.

    A book that permanently failed long ago (not in the 24h window) must not
    inflate the failure rate for recent activity.
    """
    import sqlite3
    from agentic_pipeline.health import HealthMonitor
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create 15 completed books (recent)
    for i in range(15):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_state(pid, PipelineState.HASHING)
        repo.update_state(pid, PipelineState.CLASSIFYING)
        repo.update_state(pid, PipelineState.SELECTING_STRATEGY)
        repo.update_state(pid, PipelineState.PROCESSING)
        repo.update_state(pid, PipelineState.VALIDATING)
        repo.update_state(pid, PipelineState.PENDING_APPROVAL)
        repo.update_state(pid, PipelineState.APPROVED)
        repo.update_state(pid, PipelineState.EMBEDDING)
        repo.update_state(pid, PipelineState.COMPLETE)

    # Create 4 permanently failed books, then backdate them to 48h ago.
    # If counted all-time: (4)/(15+4) = 21% > 20% → alert would fire incorrectly.
    # If time-windowed: 0 recent failures / 15 recent = 0% → no alert.
    for i in range(4):
        pid = repo.create(f"/old_fail{i}.epub", f"oldhash{i}")
        repo.update_state(pid, PipelineState.NEEDS_RETRY)
        repo.update_state(pid, PipelineState.FAILED)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE processing_pipelines SET updated_at = datetime('now', '-48 hours') WHERE state = 'failed'",
    )
    conn.commit()
    conn.close()

    monitor = HealthMonitor(db_path, alert_failure_rate=0.20)
    report = monitor.get_health()

    # Old failures outside the 24h window must not trigger a high_failure_rate alert
    assert not any(a["type"] == "high_failure_rate" for a in report["alerts"])
