# tests/test_stuck_detection.py
"""Tests for stuck detection."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

from conftest import transition_to


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_detect_stuck_pipeline(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    transition_to(repo, pid, PipelineState.PROCESSING)

    # Manually set updated_at to 2 hours ago
    conn = sqlite3.connect(db_path)
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (two_hours_ago, pid))
    conn.commit()
    conn.close()

    detector = StuckDetector(db_path)
    stuck = detector.detect()

    assert len(stuck) == 1
    assert stuck[0]["id"] == pid
    assert stuck[0]["state"] == "processing"


def test_not_stuck_if_recent(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    transition_to(repo, pid, PipelineState.PROCESSING)

    # Recently updated - should not be stuck
    detector = StuckDetector(db_path)
    stuck = detector.detect()

    assert len(stuck) == 0


def test_completed_not_flagged_as_stuck(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    transition_to(repo, pid, PipelineState.COMPLETE)

    # Set old updated_at
    conn = sqlite3.connect(db_path)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (old_time, pid))
    conn.commit()
    conn.close()

    detector = StuckDetector(db_path)
    stuck = detector.detect()

    # Should not be flagged - it's complete
    assert len(stuck) == 0


def test_default_state_thresholds():
    from agentic_pipeline.health.stuck_detector import DEFAULT_STATE_TIMEOUTS

    assert DEFAULT_STATE_TIMEOUTS["HASHING"] == 60
    assert DEFAULT_STATE_TIMEOUTS["PROCESSING"] == 900
    assert DEFAULT_STATE_TIMEOUTS["EMBEDDING"] == 600
