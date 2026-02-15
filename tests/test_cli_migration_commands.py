"""Tests for chunk-library and embed-library CLI commands."""

from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_chunk_library_dry_run():
    """chunk-library --dry-run shows what would be chunked."""
    from agentic_pipeline.cli import main

    runner = CliRunner()
    with patch("agentic_pipeline.library.migration.chunk_all_books") as mock_chunk:
        mock_chunk.return_value = {"books": 5, "chapters": 50, "chunks_created": 200}

        result = runner.invoke(main, ["chunk-library", "--dry-run"])

    assert result.exit_code == 0
    mock_chunk.assert_called_once()


def test_embed_library_dry_run():
    """embed-library --dry-run shows what would be embedded."""
    from agentic_pipeline.cli import main

    runner = CliRunner()
    with patch("agentic_pipeline.library.migration.embed_all_chunks") as mock_embed:
        mock_embed.return_value = {"chunks_embedded": 0, "total_chunks": 200, "needs_embedding": 50}

        result = runner.invoke(main, ["embed-library", "--dry-run"])

    assert result.exit_code == 0
    mock_embed.assert_called_once()
