"""Tests that pipeline_state_history is written on every state transition."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def orchestrator(db_path):
    from agentic_pipeline.orchestrator.orchestrator import Orchestrator
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig(db_path=db_path)
    return Orchestrator(config)


def _get_history(db_path, pipeline_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM pipeline_state_history WHERE pipeline_id = ? ORDER BY id",
        (pipeline_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def test_transition_writes_to_state_history(orchestrator, db_path):
    """Each _transition() call must insert a row into pipeline_state_history."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "abc123")

    orchestrator._transition(pid, PipelineState.HASHING)
    orchestrator._transition(pid, PipelineState.CLASSIFYING)

    rows = _get_history(db_path, pid)
    assert len(rows) == 2, f"Expected 2 history rows, got {len(rows)}"


def test_transition_records_from_and_to_state(orchestrator, db_path):
    """State history rows must capture from_state and to_state correctly."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "def456")

    orchestrator._transition(pid, PipelineState.HASHING)
    orchestrator._transition(pid, PipelineState.CLASSIFYING)

    rows = _get_history(db_path, pid)
    assert rows[0]["to_state"] == PipelineState.HASHING.value
    assert rows[1]["from_state"] == PipelineState.HASHING.value
    assert rows[1]["to_state"] == PipelineState.CLASSIFYING.value


def test_transition_records_duration_ms(orchestrator, db_path):
    """duration_ms must be a non-negative integer after the first transition."""
    import time
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "ghi789")

    orchestrator._transition(pid, PipelineState.HASHING)
    time.sleep(0.05)
    orchestrator._transition(pid, PipelineState.CLASSIFYING)

    rows = _get_history(db_path, pid)
    # First row has no prior state so duration_ms may be None
    assert rows[1]["duration_ms"] is not None
    assert rows[1]["duration_ms"] >= 0
