# tests/test_mcp_phase4.py
"""Tests for Phase 4 MCP tools."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_get_pipeline_health_exists(db_path):
    from agentic_pipeline.mcp_server import get_pipeline_health

    assert callable(get_pipeline_health)


def test_get_stuck_pipelines_exists(db_path):
    from agentic_pipeline.mcp_server import get_stuck_pipelines

    assert callable(get_stuck_pipelines)


def test_batch_approve_tool_exists(db_path):
    from agentic_pipeline.mcp_server import batch_approve_tool

    assert callable(batch_approve_tool)


def test_get_audit_log_exists(db_path):
    from agentic_pipeline.mcp_server import get_audit_log

    assert callable(get_audit_log)
