# tests/test_orchestrator.py
"""Tests for the Pipeline Orchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def config(db_path):
    from agentic_pipeline.config import OrchestratorConfig

    return OrchestratorConfig(
        db_path=db_path,
        book_ingestion_path=Path("/mock/path"),
        processing_timeout=10,
        embedding_timeout=5,
    )


def test_orchestrator_initializes(config):
    from agentic_pipeline.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    assert orchestrator.config == config
    assert orchestrator.shutdown_requested == False


def test_orchestrator_idempotency_skips_complete(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Create a completed pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.HASHING)
    repo.update_state(pid, PipelineState.COMPLETE)

    orchestrator = Orchestrator(config)

    # Mock file hashing to return same hash
    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "already" in result["reason"].lower()


def test_orchestrator_idempotency_skips_in_progress(db_path, config):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Create an in-progress pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.HASHING)
    repo.update_state(pid, PipelineState.PROCESSING)

    orchestrator = Orchestrator(config)

    with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
        result = orchestrator.process_one("/book.epub")

    assert result["skipped"] == True
    assert "in progress" in result["reason"].lower()
