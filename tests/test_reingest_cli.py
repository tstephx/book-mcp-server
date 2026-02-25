"""Tests for reingest CLI command --force-fallback flag."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from agentic_pipeline.cli import main


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def _insert_pipeline(db_path, book_id, source_path):
    """Insert a completed pipeline record."""
    from agentic_pipeline.db.connection import get_pipeline_db
    from agentic_pipeline.pipeline.states import PipelineState

    with get_pipeline_db(str(db_path)) as conn:
        conn.execute(
            """INSERT INTO processing_pipelines
               (id, source_path, state, content_hash)
               VALUES (?, ?, ?, ?)""",
            (book_id, source_path, PipelineState.COMPLETE.value, "abc123"),
        )
        conn.commit()


def test_reingest_force_fallback_passes_flag_to_orchestrator(db_path, tmp_path):
    """reingest --force-fallback must pass force_fallback=True to _process_book."""
    source_file = tmp_path / "test.epub"
    source_file.write_bytes(b"fake epub")
    book_id = "test-book-id"
    _insert_pipeline(db_path, book_id, str(source_file))

    runner = CliRunner()
    mock_result = {"state": "complete"}

    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_path)), \
         patch("agentic_pipeline.orchestrator.orchestrator.Orchestrator._process_book",
               return_value=mock_result) as mock_process:
        result = runner.invoke(main, ["reingest", book_id, "--force-fallback"])

    assert result.exit_code == 0, result.output
    mock_process.assert_called_once()
    _, kwargs = mock_process.call_args
    assert kwargs.get("force_fallback") is True, (
        f"Expected force_fallback=True, got: {mock_process.call_args}"
    )


def test_reingest_without_force_fallback_does_not_pass_flag(db_path, tmp_path):
    """reingest without --force-fallback must pass force_fallback=False (default)."""
    source_file = tmp_path / "test.epub"
    source_file.write_bytes(b"fake epub")
    book_id = "test-book-id-2"
    _insert_pipeline(db_path, book_id, str(source_file))

    runner = CliRunner()
    mock_result = {"state": "complete"}

    with patch("agentic_pipeline.db.config.get_db_path", return_value=str(db_path)), \
         patch("agentic_pipeline.orchestrator.orchestrator.Orchestrator._process_book",
               return_value=mock_result) as mock_process:
        result = runner.invoke(main, ["reingest", book_id])

    assert result.exit_code == 0, result.output
    mock_process.assert_called_once()
    _, kwargs = mock_process.call_args
    assert kwargs.get("force_fallback") is False or kwargs.get("force_fallback") is None
