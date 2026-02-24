"""Tests for the spot-check CLI command."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
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


@pytest.fixture
def runner():
    return CliRunner()


def _patch_db(db_path):
    """Patch get_db_path at the module it's imported from inside the CLI."""
    return patch("agentic_pipeline.db.config.get_db_path", return_value=db_path)


def _patch_candidates(candidates):
    """Patch the sampled list used by the interactive review path."""
    return patch(
        "agentic_pipeline.autonomy.spot_check.SpotCheckManager.select_for_review",
        return_value=candidates,
    )


def _patch_all_unreviewed(candidates):
    """Patch the full unsampled list used by --list."""
    return patch(
        "agentic_pipeline.autonomy.spot_check.SpotCheckManager.get_all_unreviewed",
        return_value=candidates,
    )


def _patch_title(title="Test Book"):
    return patch("agentic_pipeline.cli._get_book_title", return_value=title)


def _patch_submit(submitted):
    def _record(**kwargs):
        submitted.append(kwargs)
    return patch(
        "agentic_pipeline.autonomy.spot_check.SpotCheckManager.submit_result",
        side_effect=_record,
    )


MOCK_BOOK = {
    "book_id": "abc12345-0000-0000-0000-000000000001",
    "original_book_type": "technical",
    "original_confidence": 0.85,
    "created_at": "2026-02-20T10:00:00",
}


def test_spot_check_no_candidates(runner, db_path):
    """When no auto-approved books exist, interactive mode prints a friendly message."""
    with _patch_db(db_path), _patch_candidates([]):
        result = runner.invoke(main, ["spot-check"])

    assert result.exit_code == 0
    assert "No spot-checks pending" in result.output


def test_spot_check_list_no_candidates(runner, db_path):
    """--list with no candidates prints a friendly message."""
    with _patch_db(db_path), _patch_all_unreviewed([]):
        result = runner.invoke(main, ["spot-check", "--list"])

    assert result.exit_code == 0
    assert "No spot-checks pending" in result.output


def test_spot_check_list_shows_full_unsampled_count(runner, db_path):
    """--list uses the full unsampled list, not the 10% sample."""
    # 10 books in the full list — if sampling were applied, fewer would show
    all_books = [
        dict(MOCK_BOOK, book_id=f"abc12345-0000-0000-0000-00000000000{i}")
        for i in range(10)
    ]
    with _patch_db(db_path), _patch_all_unreviewed(all_books), _patch_title("Test Book"):
        result = runner.invoke(main, ["spot-check", "--list"])

    assert result.exit_code == 0
    assert "10 books" in result.output


def test_spot_check_list_shows_table(runner, db_path):
    """--list with candidates shows a table with title and type."""
    with _patch_db(db_path), _patch_all_unreviewed([MOCK_BOOK]), _patch_title("Test Book Title"):
        result = runner.invoke(main, ["spot-check", "--list"])

    assert result.exit_code == 0
    assert "Spot-Check" in result.output
    assert "technical" in result.output
    assert "Test Book Title" in result.output


def test_spot_check_interactive_correct(runner, db_path):
    """Answering 'y' submits classification_correct=True, quality_acceptable=True."""
    submitted = []
    with _patch_db(db_path), _patch_candidates([MOCK_BOOK]), _patch_title(), _patch_submit(submitted):
        result = runner.invoke(main, ["spot-check"], input="y\n")

    assert result.exit_code == 0
    assert len(submitted) == 1
    assert submitted[0]["classification_correct"] is True
    assert submitted[0]["quality_acceptable"] is True
    assert "correct" in result.output


def test_spot_check_interactive_incorrect(runner, db_path):
    """Answering 'n' then providing details submits an issue."""
    submitted = []
    # n -> classification correct? n -> quality ok? y -> notes: "wrong type"
    with _patch_db(db_path), _patch_candidates([MOCK_BOOK]), _patch_title(), _patch_submit(submitted):
        result = runner.invoke(main, ["spot-check"], input="n\nn\ny\nwrong type\n")

    assert result.exit_code == 0
    assert len(submitted) == 1
    assert submitted[0]["classification_correct"] is False
    assert submitted[0]["quality_acceptable"] is True
    assert submitted[0]["notes"] == "wrong type"
    assert "issue flagged" in result.output


def test_spot_check_interactive_quit(runner, db_path):
    """Answering 'q' exits early and prints a summary."""
    books = [dict(MOCK_BOOK, book_id=f"abc12345-0000-0000-0000-00000000000{i}") for i in range(1, 4)]
    with _patch_db(db_path), _patch_candidates(books), _patch_title():
        result = runner.invoke(main, ["spot-check"], input="q\n")

    assert result.exit_code == 0
    assert "stopped early" in result.output


def test_spot_check_interactive_skip(runner, db_path):
    """Answering 's' skips a book without recording a result."""
    submitted = []
    with _patch_db(db_path), _patch_candidates([MOCK_BOOK]), _patch_title(), _patch_submit(submitted):
        result = runner.invoke(main, ["spot-check"], input="s\n")

    assert result.exit_code == 0
    assert len(submitted) == 0
    assert "Skipped" in result.output


def test_spot_check_summary_shown_after_full_review(runner, db_path):
    """Session summary is printed after reviewing all books."""
    submitted = []
    with _patch_db(db_path), _patch_candidates([MOCK_BOOK]), _patch_title(), _patch_submit(submitted):
        result = runner.invoke(main, ["spot-check"], input="y\n")

    assert "Session summary" in result.output
    assert "1 reviewed" in result.output
