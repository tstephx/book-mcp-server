# tests/test_mcp_phase5.py
"""Tests for Phase 5 MCP tools."""

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


def test_get_autonomy_status_exists(db_path):
    from agentic_pipeline.mcp_server import get_autonomy_status

    assert callable(get_autonomy_status)


def test_set_autonomy_mode_exists(db_path):
    from agentic_pipeline.mcp_server import set_autonomy_mode

    assert callable(set_autonomy_mode)


def test_activate_escape_hatch_exists(db_path):
    from agentic_pipeline.mcp_server import activate_escape_hatch_tool

    assert callable(activate_escape_hatch_tool)


def test_get_autonomy_readiness_exists(db_path):
    from agentic_pipeline.mcp_server import get_autonomy_readiness

    assert callable(get_autonomy_readiness)
