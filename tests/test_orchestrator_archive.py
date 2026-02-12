# tests/test_orchestrator_archive.py
"""Tests for auto-archive of processed book files."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


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
        processing_timeout=10,
        embedding_timeout=5,
    )


# --- Standalone _archive_source_file tests (approval/actions.py) ---


def test_archive_moves_file_to_processed_dir(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"fake epub content")

    processed_dir = tmp_path / "processed"

    result = _archive_source_file(str(src_file), processed_dir=processed_dir)

    assert result == "book.epub"
    assert not src_file.exists()
    assert (processed_dir / "book.epub").exists()
    assert (processed_dir / "book.epub").read_bytes() == b"fake epub content"


def test_archive_creates_processed_dir(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "deep" / "nested" / "processed"

    _archive_source_file(str(src_file), processed_dir=processed_dir)

    assert processed_dir.exists()
    assert (processed_dir / "book.epub").exists()


def test_archive_noop_when_no_processed_dir(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    # No processed_dir argument, no PROCESSED_DIR env var
    with patch.dict("os.environ", {}, clear=True):
        result = _archive_source_file(str(src_file))

    assert result is None
    # File should remain untouched
    assert src_file.exists()


def test_archive_handles_missing_source_file(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    processed_dir = tmp_path / "processed"

    # Should not raise — just logs and returns None
    result = _archive_source_file(str(tmp_path / "nonexistent.epub"), processed_dir=processed_dir)
    assert result is None


def test_archive_handles_name_collision(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Pre-create a file with the same name
    (processed_dir / "book.epub").write_bytes(b"existing")

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"new content")

    result = _archive_source_file(str(src_file), processed_dir=processed_dir)

    assert result == "book_1.epub"
    assert not src_file.exists()
    # Original should be untouched
    assert (processed_dir / "book.epub").read_bytes() == b"existing"
    # New file should get a counter suffix
    assert (processed_dir / "book_1.epub").exists()
    assert (processed_dir / "book_1.epub").read_bytes() == b"new content"


def test_archive_handles_multiple_collisions(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Pre-create files with the name and first counter
    (processed_dir / "book.epub").write_bytes(b"v0")
    (processed_dir / "book_1.epub").write_bytes(b"v1")

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"v2")

    result = _archive_source_file(str(src_file), processed_dir=processed_dir)

    assert result == "book_2.epub"
    assert (processed_dir / "book_2.epub").exists()
    assert (processed_dir / "book_2.epub").read_bytes() == b"v2"


def test_archive_logs_error_on_move_failure(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "processed"

    with patch("agentic_pipeline.approval.actions.shutil.move", side_effect=OSError("disk full")):
        result = _archive_source_file(str(src_file), processed_dir=processed_dir)

    assert result is None
    # Source file should still exist (move failed)
    assert src_file.exists()


def test_archive_uses_env_var_fallback(tmp_path):
    from agentic_pipeline.approval.actions import _archive_source_file

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "processed"

    with patch.dict("os.environ", {"PROCESSED_DIR": str(processed_dir)}):
        result = _archive_source_file(str(src_file))

    assert result == "book.epub"
    assert not src_file.exists()
    assert (processed_dir / "book.epub").exists()


# --- Orchestrator scan exclusion test ---


def test_scan_skips_files_in_processed_dir(db_path, config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Book in watch dir (should be detected)
    (tmp_path / "new_book.epub").write_bytes(b"new book")
    # Book in processed dir (should be skipped)
    (processed_dir / "old_book.epub").write_bytes(b"already processed")

    config.watch_dir = tmp_path
    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)
    detected = orchestrator._scan_watch_dir()

    assert detected == 1


# --- Integration: _complete_approved calls archive ---


def test_complete_approved_archives_on_success(db_path, tmp_path):
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from tests.conftest import transition_to

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "processed"

    # Create a real pipeline record and walk it to APPROVED
    repo = PipelineRepository(db_path)
    pipeline_id = repo.create(str(src_file), "abc123hash")
    transition_to(repo, pipeline_id, PipelineState.APPROVED)

    pipeline = repo.get(pipeline_id)

    with patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter") as MockAdapter:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.chapters_processed = 5
        MockAdapter.return_value.generate_embeddings.return_value = mock_result

        with patch.dict("os.environ", {"PROCESSED_DIR": str(processed_dir)}):
            result = _complete_approved(db_path, pipeline_id, pipeline)

    assert result["state"] == "complete"
    assert not src_file.exists()
    assert (processed_dir / "book.epub").exists()


def test_complete_approved_does_not_archive_on_failure(db_path, tmp_path):
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from tests.conftest import transition_to

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "processed"

    # Create a real pipeline record and walk it to APPROVED
    repo = PipelineRepository(db_path)
    pipeline_id = repo.create(str(src_file), "def456hash")
    transition_to(repo, pipeline_id, PipelineState.APPROVED)

    pipeline = repo.get(pipeline_id)

    with patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter") as MockAdapter:
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "embedding failed"
        MockAdapter.return_value.generate_embeddings.return_value = mock_result

        with patch.dict("os.environ", {"PROCESSED_DIR": str(processed_dir)}):
            result = _complete_approved(db_path, pipeline_id, pipeline)

    assert result["state"] == "needs_retry"
    # File should NOT have been archived
    assert src_file.exists()
    assert not processed_dir.exists()
