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
