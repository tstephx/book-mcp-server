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
    assert args is not None and type(None) in args, f"book_type annotation {annotation} is not Optional[str]"


# ---------------------------------------------------------------------------
# C3: reingest_book_tool must use a public Orchestrator method, not _process_book
# ---------------------------------------------------------------------------


def test_orchestrator_exposes_reprocess_existing_as_public_method():
    """Orchestrator must have a public reprocess_existing() method."""
    from agentic_pipeline.orchestrator import Orchestrator

    assert hasattr(Orchestrator, "reprocess_existing"), (
        "Orchestrator has no public reprocess_existing() method. "
        "reingest_book_tool must not call _process_book directly."
    )
    assert not hasattr(Orchestrator, "__reprocess_existing"), "Should be public, not mangled"


def test_reingest_book_tool_does_not_call_private_process_book(db_path):
    """reingest_book_tool must route through reprocess_existing(), not _process_book."""
    import inspect
    import agentic_pipeline.mcp_server as mcp_mod

    source = inspect.getsource(mcp_mod.reingest_book_tool)
    assert "._process_book(" not in source, (
        "reingest_book_tool still calls the private _process_book() directly. "
        "Use orchestrator.reprocess_existing() instead."
    )


def test_reprocess_existing_returns_pipeline_result(db_path, tmp_path):
    """reprocess_existing() drives the pipeline for an existing record and returns result."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.db.pipelines import PipelineRepository
    from unittest.mock import MagicMock, patch

    # Create a pipeline record (simulating what prepare_reingest would do)
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "testhash123")

    config = OrchestratorConfig(db_path=db_path)
    orchestrator = Orchestrator(config)

    mock_result = {
        "pipeline_id": pid,
        "state": "pending_approval",
        "book_type": "technical_tutorial",
        "confidence": 0.9,
    }

    with patch.object(orchestrator, "_process_book", return_value=mock_result) as mock_pb:
        result = orchestrator.reprocess_existing(pid, "/book.epub", "testhash123")

    mock_pb.assert_called_once_with(pid, "/book.epub", "testhash123", force_fallback=False)
    assert result["pipeline_id"] == pid
