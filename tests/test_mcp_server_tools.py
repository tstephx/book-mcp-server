"""Tests for MCP server tool layer (mcp_server.py).

Covers: error handling, input validation, response shape consistency.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# C4: set_autonomy_mode — invalid mode must return error, not raise
# ---------------------------------------------------------------------------

def test_set_autonomy_mode_invalid_returns_error_not_exception(db_path):
    """set_autonomy_mode('invalid') must return {"error": ...} not raise ValueError."""
    from agentic_pipeline.mcp_server import set_autonomy_mode

    with patch("agentic_pipeline.mcp_server.get_db_path", return_value=db_path):
        result = set_autonomy_mode("invalid_mode")

    assert "error" in result
    assert result.get("success") is not True


def test_set_autonomy_mode_valid_returns_success(db_path):
    """set_autonomy_mode with a valid mode returns {"success": True, "mode": ...}."""
    from agentic_pipeline.mcp_server import set_autonomy_mode

    with patch("agentic_pipeline.mcp_server.get_db_path", return_value=db_path):
        result = set_autonomy_mode("partial")

    assert result.get("success") is True
    assert result["mode"] == "partial"


# ---------------------------------------------------------------------------
# C1: process_book — path validation
# ---------------------------------------------------------------------------

def test_process_book_rejects_disallowed_extension():
    """process_book must reject files that are not .epub or .pdf."""
    from agentic_pipeline.mcp_server import process_book

    result = process_book("/tmp/exploit.sh")

    assert "error" in result
    assert result.get("success") is not True


def test_process_book_rejects_path_traversal():
    """process_book must reject paths containing ../ traversal sequences."""
    from agentic_pipeline.mcp_server import process_book

    result = process_book("../../../../etc/passwd")

    assert "error" in result
    assert result.get("success") is not True


def test_process_book_rejects_empty_path():
    """process_book must reject an empty string path."""
    from agentic_pipeline.mcp_server import process_book

    result = process_book("")

    assert "error" in result


# ---------------------------------------------------------------------------
# W8: missing tool registrations — functions must exist in mcp_server module
# ---------------------------------------------------------------------------

def test_backfill_library_is_importable():
    """backfill_library must be importable from mcp_server (not missing)."""
    from agentic_pipeline.mcp_server import backfill_library  # noqa: F401


def test_validate_library_is_importable():
    """validate_library must be importable from mcp_server."""
    from agentic_pipeline.mcp_server import validate_library  # noqa: F401


def test_reingest_book_tool_is_importable():
    """reingest_book_tool must be importable from mcp_server."""
    from agentic_pipeline.mcp_server import reingest_book_tool  # noqa: F401


def test_backfill_validate_reingest_registered_in_wrapper():
    """backfill, validate_library, and reingest must be registered in agentic_mcp_server."""
    import agentic_mcp_server

    tool_names = list(agentic_mcp_server.mcp._tool_manager._tools.keys())
    assert "backfill" in tool_names, f"backfill not registered. Tools: {tool_names}"
    assert "validate_library" in tool_names, f"validate_library not registered. Tools: {tool_names}"
    assert "reingest" in tool_names, f"reingest not registered. Tools: {tool_names}"


# ---------------------------------------------------------------------------
# W2: type annotations — Optional parameters must be Optional
# ---------------------------------------------------------------------------

def test_batch_approve_accepts_none_confidence():
    """batch_approve_tool must accept None for min_confidence (Optional[float])."""
    from agentic_pipeline.mcp_server import batch_approve_tool
    import inspect

    sig = inspect.signature(batch_approve_tool)
    param = sig.parameters["min_confidence"]
    # Should accept None — the annotation should be Optional or the default should be None
    assert param.default is None


def test_batch_approve_accepts_none_book_type():
    """batch_approve_tool must accept None for book_type (Optional[str])."""
    from agentic_pipeline.mcp_server import batch_approve_tool
    import inspect

    sig = inspect.signature(batch_approve_tool)
    param = sig.parameters["book_type"]
    annotation = param.annotation
    # Must be Optional[str] (i.e., Union[str, None] or str | None), not bare str
    import typing
    args = getattr(annotation, "__args__", None)
    assert args is not None and type(None) in args, (
        f"book_type annotation {annotation} is not Optional[str]"
    )
