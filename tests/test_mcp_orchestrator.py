# tests/test_mcp_orchestrator.py
"""Tests for orchestrator MCP tools."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def db_path(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_process_book_tool_exists(db_path):
    from agentic_pipeline.mcp_server import process_book

    assert callable(process_book)


def test_get_pipeline_status_tool_exists(db_path):
    from agentic_pipeline.mcp_server import get_pipeline_status

    assert callable(get_pipeline_status)
