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


def test_archive_moves_file_to_processed_dir(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"fake epub content")

    processed_dir = tmp_path / "processed"
    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)
    orchestrator._archive_source_file("pipe-1", str(src_file))

    assert not src_file.exists()
    assert (processed_dir / "book.epub").exists()
    assert (processed_dir / "book.epub").read_bytes() == b"fake epub content"


def test_archive_creates_processed_dir(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "deep" / "nested" / "processed"
    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)
    orchestrator._archive_source_file("pipe-1", str(src_file))

    assert processed_dir.exists()
    assert (processed_dir / "book.epub").exists()


def test_archive_noop_when_processed_dir_not_set(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    config.processed_dir = None

    orchestrator = Orchestrator(config)
    orchestrator._archive_source_file("pipe-1", str(src_file))

    # File should remain untouched
    assert src_file.exists()


def test_archive_handles_missing_source_file(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    config.processed_dir = tmp_path / "processed"

    orchestrator = Orchestrator(config)
    # Should not raise — just logs and returns
    orchestrator._archive_source_file("pipe-1", str(tmp_path / "nonexistent.epub"))


def test_archive_handles_name_collision(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Pre-create a file with the same name
    (processed_dir / "book.epub").write_bytes(b"existing")

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"new content")

    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)
    orchestrator._archive_source_file("pipe-1", str(src_file))

    assert not src_file.exists()
    # Original should be untouched
    assert (processed_dir / "book.epub").read_bytes() == b"existing"
    # New file should get a counter suffix
    assert (processed_dir / "book_1.epub").exists()
    assert (processed_dir / "book_1.epub").read_bytes() == b"new content"


def test_archive_handles_multiple_collisions(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Pre-create files with the name and first counter
    (processed_dir / "book.epub").write_bytes(b"v0")
    (processed_dir / "book_1.epub").write_bytes(b"v1")

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"v2")

    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)
    orchestrator._archive_source_file("pipe-1", str(src_file))

    assert (processed_dir / "book_2.epub").exists()
    assert (processed_dir / "book_2.epub").read_bytes() == b"v2"


def test_archive_logs_error_on_move_failure(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    config.processed_dir = tmp_path / "processed"

    orchestrator = Orchestrator(config)

    with patch("agentic_pipeline.orchestrator.orchestrator.shutil.move", side_effect=OSError("disk full")):
        # Should not raise
        orchestrator._archive_source_file("pipe-1", str(src_file))

    # Source file should still exist (move failed)
    assert src_file.exists()


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


def test_archive_not_called_when_completion_fails(config, tmp_path):
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.pipeline.states import PipelineState

    src_file = tmp_path / "book.epub"
    src_file.write_bytes(b"content")

    processed_dir = tmp_path / "processed"
    config.processed_dir = processed_dir

    orchestrator = Orchestrator(config)

    # Mock _complete_approved to return a non-COMPLETE state
    with patch.object(orchestrator, '_complete_approved', return_value={"state": PipelineState.NEEDS_RETRY.value}):
        with patch.object(orchestrator, '_archive_source_file') as mock_archive:
            with patch.object(orchestrator, '_run_processing', return_value={
                "book_id": "test", "quality_score": 85, "detection_confidence": 0.9,
                "detection_method": "mock", "needs_review": False, "warnings": [],
                "chapter_count": 10, "word_count": 50000, "llm_fallback_used": False,
            }):
                with patch.object(orchestrator, '_compute_hash', return_value="hash123"):
                    with patch.object(orchestrator, '_extract_sample', return_value="text"):
                        from agentic_pipeline.agents.classifier_types import BookProfile, BookType
                        mock_profile = BookProfile(
                            book_type=BookType.TECHNICAL_TUTORIAL,
                            confidence=0.9,
                            suggested_tags=["python"],
                            reasoning="Test",
                        )
                        orchestrator.classifier = MagicMock()
                        orchestrator.classifier.classify.return_value = mock_profile

                        result = orchestrator.process_one(str(src_file))

            # Archive should NOT have been called since state != COMPLETE
            mock_archive.assert_not_called()
