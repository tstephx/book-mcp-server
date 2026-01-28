"""Tests for MCP tools."""

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


@pytest.fixture
def setup_pending_books(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/test/book.epub", "testhash")
    repo.update_book_profile(pid, {
        "book_type": "technical_tutorial",
        "confidence": 0.92
    })
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)
    return pid


def test_review_pending_books_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import review_pending_books

    result = review_pending_books(str(db_path))

    assert result["pending_count"] == 1
    assert len(result["books"]) == 1
    assert result["books"][0]["book_type"] == "technical_tutorial"


def test_approve_book_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import approve_book_tool

    pid = setup_pending_books
    result = approve_book_tool(str(db_path), pid)

    assert result["success"] is True


def test_reject_book_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import reject_book_tool

    pid = setup_pending_books
    result = reject_book_tool(str(db_path), pid, "Test rejection")

    assert result["success"] is True
    assert result["state"] == "rejected"
